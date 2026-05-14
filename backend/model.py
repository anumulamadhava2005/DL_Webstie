# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Dropout2d(drop),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
    def forward(self, x): return self.block(x)


class AttentionGate(nn.Module):
    """Standard additive attention gate (Oktay et al. 2018)."""
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg = nn.Conv2d(g_ch,    inter_ch, 1, bias=False)
        self.Wx = nn.Conv2d(x_ch,    inter_ch, 1, bias=False)
        self.psi= nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.Sigmoid())
        self.bn = nn.BatchNorm2d(inter_ch)

    def forward(self, g, x):
        # g: gating signal from coarser scale; x: skip connection
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        att  = self.psi(F.relu(self.bn(self.Wg(g_up) + self.Wx(x)), inplace=True))
        return x * att


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: text → (γ, β) scale/shift of feature maps."""
    def __init__(self, text_dim, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, feat_dim * 2),
            nn.SiLU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )

    def forward(self, x, t):
        out = self.net(t)                              # (B, 2*C)
        g, b = out.chunk(2, dim=-1)                   # each (B, C)
        g = g.view(-1, x.size(1), 1, 1)
        b = b.view(-1, x.size(1), 1, 1)
        return (1 + g) * x + b                        # learnable scale + shift


class TextGuidedAttentionUNet(nn.Module):
    """
    Attention UNet with:
    · 4-level encoder / decoder (base_ch = 64 → 128 → 256 → 512 → 1024)
    · Attention gates on every skip connection
    · Multi-scale FiLM text conditioning (bottleneck + all decoder levels)
    · Deep supervision outputs at all 4 decoder scales
    """
    def __init__(self, in_ch=1, text_dim=512, base=64, use_text=True):
        super().__init__()
        self.use_text = use_text
        c = base

        # ── encoder ──
        self.enc1 = ConvBlock(in_ch, c)
        self.enc2 = ConvBlock(c,   c*2)
        self.enc3 = ConvBlock(c*2, c*4)
        self.enc4 = ConvBlock(c*4, c*8)
        self.bnck = ConvBlock(c*8, c*16)
        self.pool = nn.MaxPool2d(2)

        # ── multi-scale FiLM (text → every decoder level) ──
        if use_text:
            self.film_b  = FiLMLayer(text_dim, c*16)
            self.film_d4 = FiLMLayer(text_dim, c*8)
            self.film_d3 = FiLMLayer(text_dim, c*4)
            self.film_d2 = FiLMLayer(text_dim, c*2)
            self.film_d1 = FiLMLayer(text_dim, c)

        # ── attention gates ──
        self.att4 = AttentionGate(c*16, c*8,  c*4)
        self.att3 = AttentionGate(c*8,  c*4,  c*2)
        self.att2 = AttentionGate(c*4,  c*2,  c)
        self.att1 = AttentionGate(c*2,  c,    c//2)

        # ── decoder ──
        self.up4  = nn.ConvTranspose2d(c*16, c*8,  2, stride=2)
        self.dec4 = ConvBlock(c*16,  c*8)
        self.up3  = nn.ConvTranspose2d(c*8,  c*4,  2, stride=2)
        self.dec3 = ConvBlock(c*8,   c*4)
        self.up2  = nn.ConvTranspose2d(c*4,  c*2,  2, stride=2)
        self.dec2 = ConvBlock(c*4,   c*2)
        self.up1  = nn.ConvTranspose2d(c*2,  c,    2, stride=2)
        self.dec1 = ConvBlock(c*2,   c)

        # ── deep supervision heads ──
        self.ds4 = nn.Conv2d(c*8, 1, 1)
        self.ds3 = nn.Conv2d(c*4, 1, 1)
        self.ds2 = nn.Conv2d(c*2, 1, 1)
        self.out = nn.Conv2d(c,   1, 1)      # main output

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x, tf=None):
        # ── encode ──
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bnck(self.pool(e4))

        # ── text conditioning ──
        if self.use_text and tf is not None:
            b = self.film_b(b, tf)

        # ── decode with attention + FiLM ──
        d4 = self.dec4(torch.cat([self.up4(b),  self.att4(b,  e4)], 1))
        if self.use_text and tf is not None:
            d4 = self.film_d4(d4, tf)

        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(d4, e3)], 1))
        if self.use_text and tf is not None:
            d3 = self.film_d3(d3, tf)

        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(d3, e2)], 1))
        if self.use_text and tf is not None:
            d2 = self.film_d2(d2, tf)

        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(d2, e1)], 1))
        if self.use_text and tf is not None:
            d1 = self.film_d1(d1, tf)

        # Return list of logits: [main, ds4, ds3, ds2]
        return [self.out(d1), self.ds4(d4), self.ds3(d3), self.ds2(d2)]

class GradCAM:
    def __init__(self, model):
        self.model = model
        self._acts = self._grads = None
        target = model.bnck.block[-1]          # last BN/GELU of bottleneck
        self._h  = [
            target.register_forward_hook(lambda m,i,o: setattr(self,'_acts',o.detach())),
            target.register_full_backward_hook(lambda m,gi,go: setattr(self,'_grads',go[0].detach())),
        ]

    def generate(self, img_t, tf):
        self.model.zero_grad()
        out = self.model(img_t, tf)
        out[0].mean().backward()              # backprop through main output only
        w   = self._grads.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, img_t.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.sigmoid(out[0]).detach()

    def remove(self):
        for h in self._h: h.remove()
