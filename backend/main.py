import os
import io
import base64
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager
import uvicorn
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

from model import TextGuidedAttentionUNet, GradCAM
import matplotlib.cm as cm

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models lazily to avoid startup crashes if files are missing
model_no_text = None
model_with_text = None
clip_model = None
tokenizer = None

def init_models():
    global model_no_text, model_with_text, clip_model, tokenizer
    base_dir = os.path.join(os.path.dirname(__file__), "weights")
    
    # Load model without text
    if model_no_text is None:
        path = os.path.join(base_dir, "best_without_text.pth")
        if os.path.exists(path):
            print("Loading best_without_text.pth...")
            model_no_text = TextGuidedAttentionUNet(in_ch=1, base=64, use_text=False)
            sd = torch.load(path, map_location=DEVICE)
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            model_no_text.load_state_dict(sd)
            model_no_text.to(DEVICE)
            model_no_text.eval()
            print("Loaded best_without_text.pth")
    
    # Load model with text
    if model_with_text is None:
        path = os.path.join(base_dir, "best_with_text.pth")
        if os.path.exists(path):
            print("Loading best_with_text.pth...")
            model_with_text = TextGuidedAttentionUNet(in_ch=1, base=64, use_text=True, text_dim=512)
            sd = torch.load(path, map_location=DEVICE)
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            model_with_text.load_state_dict(sd)
            model_with_text.to(DEVICE)
            model_with_text.eval()
            print("Loaded best_with_text.pth")
            
            import open_clip
            print("Loading OpenCLIP...")
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            clip_model = clip_model.to(DEVICE).eval()
            for p in clip_model.parameters():
                p.requires_grad_(False)
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            print("Loaded OpenCLIP.")


def normalize_slice(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr)
    lo  = np.percentile(arr, 1)
    hi  = np.percentile(arr, 99)
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr.astype(np.float32)

def process_npy_file(arr: np.ndarray):
    if arr.ndim == 3:
        idx = arr.shape[2] // 2
        sl = arr[:, :, idx]
    elif arr.ndim == 2:
        sl = arr
    else:
        raise ValueError("Unexpected image shape")
    return normalize_slice(sl)

@torch.no_grad()
def encode_text(text: str) -> torch.Tensor:
    if clip_model is None or tokenizer is None:
        raise ValueError("CLIP model not loaded")
    tokens = tokenizer([text]).to(DEVICE)
    return F.normalize(clip_model.encode_text(tokens).float(), dim=-1)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form("without_text"), # 'without_text' or 'with_text'
    text: str = Form("")
):
    try:
        content = await file.read()
        
        # Load NPY
        arr = np.load(io.BytesIO(content))
        img_slice = process_npy_file(arr)
        
        # Prepare input tensor
        img_t = torch.from_numpy(img_slice).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        
        # Original size for resizing back later
        orig_h, orig_w = img_t.shape[2], img_t.shape[3]
        
        # Resize to 256x256 as expected by the model
        img_t = F.interpolate(img_t, size=(256, 256), mode='bilinear', align_corners=False)
        img_t = img_t.to(DEVICE)
        img_t.requires_grad = True # Needed for Grad-CAM
        
        # Inference and Grad-CAM
        model = model_with_text if model_type == "with_text" else model_no_text
        if model is None:
            raise HTTPException(status_code=500, detail=f"Model {model_type} not found.")
        
        tf = None
        if model_type == "with_text":
            tf = encode_text(text).to(DEVICE)
        
        gcam = GradCAM(model)
        heatmap, main_pred = gcam.generate(img_t, tf)
        gcam.remove()
        
        # Resize mask back to original size
        main_pred_resized = F.interpolate(main_pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        mask_prob = main_pred_resized.squeeze().detach().cpu().numpy()
        mask_bin = (mask_prob > 0.5).astype(np.uint8)
        
        # 1. Original Image
        img_uint8 = (img_slice * 255).astype(np.uint8)
        orig_pil = Image.fromarray(img_uint8).convert("RGB")
        
        # 2. Predicted Mask
        mask_pil = Image.fromarray(mask_bin * 255).convert("L")
        
        # 3. Overlay
        overlay_pil = orig_pil.copy()
        mask_rgb = np.zeros_like(np.array(orig_pil))
        mask_rgb[mask_bin == 1] = [99, 102, 241] # Indigo color
        mask_overlay_pil = Image.fromarray(mask_rgb).convert("RGB")
        overlay_pil = Image.blend(orig_pil, mask_overlay_pil, alpha=0.5)
        
        # 4. Detected Region (Crop)
        y_indices, x_indices = np.where(mask_bin > 0)
        if len(y_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            # Add some padding
            pad = 20
            y_min = max(0, y_min - pad)
            y_max = min(orig_h, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(orig_w, x_max + pad)
            crop_pil = orig_pil.crop((x_min, y_min, x_max, y_max))
        else:
            crop_pil = orig_pil # No tumor detected, return original
            
        # 5. Grad-CAM heatmap
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.BILINEAR)
        heatmap_np = np.array(heatmap_resized)
        color_heatmap = cm.jet(heatmap_np / 255.0)[:, :, :3] # Use jet colormap
        color_heatmap = (color_heatmap * 255).astype(np.uint8)
        grad_cam_pil = Image.blend(orig_pil, Image.fromarray(color_heatmap), alpha=0.5)
        
        def pil_to_b64(pil_img):
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        
        return {
            "original_image": pil_to_b64(orig_pil),
            "mask_image": pil_to_b64(mask_pil),
            "overlay_image": pil_to_b64(overlay_pil),
            "crop_image": pil_to_b64(crop_pil),
            "grad_cam_image": pil_to_b64(grad_cam_pil)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False if os.environ.get("RENDER") else True)
