import { useState, useRef } from 'react';
import { 
  Upload, 
  File, 
  Activity, 
  Search, 
  Image as ImageIcon, 
  Maximize, 
  Brain, 
  ShieldAlert, 
  Sparkles, 
  Target, 
  ChevronRight,
  AlertCircle
} from 'lucide-react';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [modelType, setModelType] = useState('without_text');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [zoomImage, setZoomImage] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.name.endsWith('.npy')) {
      setFile(selected);
      setError(null);
    } else {
      setFile(null);
      setError('Invalid format. Please select an NPY file.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return setError('Please upload an MRI scan');
    
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);
    if (modelType === 'with_text') {
      formData.append('text', text);
    }

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Processing failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="navbar">
        <div className="navbar-brand">
          <Activity className="logo-icon" />
          <h1>NeuroScan Pro</h1>
        </div>
      </header>

      <main className="main-layout">
        <aside className="sidebar">
          <div>
            <h2 className="section-title">Scan Input</h2>
            <div 
              className={`upload-area ${file ? 'active' : ''}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".npy" 
                hidden
              />
              <Upload className="upload-icon" />
              <p className="upload-text">{file ? file.name : 'Upload MRI Scan'}</p>
              <p className="upload-subtext">Drag & drop .npy file here</p>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="label">Segmentation Model</label>
              <div className="select-grid">
                <button
                  type="button"
                  onClick={() => setModelType('without_text')}
                  className={`select-btn ${modelType === 'without_text' ? 'active' : ''}`}
                >
                  Standard
                </button>
                <button
                  type="button"
                  onClick={() => setModelType('with_text')}
                  className={`select-btn ${modelType === 'with_text' ? 'active' : ''}`}
                >
                  Text-Guided
                </button>
              </div>
            </div>

            {modelType === 'with_text' && (
              <div className="form-group">
                <label className="label">Clinical Context</label>
                <textarea 
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="e.g. FLAIR hyperintensity in the temporal lobe..."
                  className="textarea"
                />
              </div>
            )}

            {error && (
              <div className="error-box">
                <AlertCircle size={16} style={{ marginBottom: '4px' }} />
                <p>{error}</p>
              </div>
            )}

            <button 
              type="submit" 
              disabled={loading || !file}
              className="btn-primary"
            >
              {loading ? 'Processing...' : 'Run Analysis'}
              {!loading && <ChevronRight size={18} />}
            </button>
          </form>
          
          <div style={{ marginTop: 'auto', paddingTop: '20px', borderTop: '1px solid var(--border)' }}>
             <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
               NeuroScan Pro v2.5.0<br />
               System Status: Ready
             </p>
          </div>
        </aside>

        <section className="content">
          {result && (
            <div className="view-tabs">
              <div 
                className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`}
                onClick={() => setActiveTab('dashboard')}
              >
                Dashboard
              </div>
              <div 
                className={`tab ${activeTab === 'detailed' ? 'active' : ''}`}
                onClick={() => setActiveTab('detailed')}
              >
                Detailed View
              </div>
            </div>
          )}

          {!result && !loading && (
            <div className="empty-state">
              <ImageIcon size={48} />
              <p>Upload a scan to begin clinical analysis</p>
            </div>
          )}

          {loading && (
            <div className="loading-container">
              <div className="spinner"></div>
              <p style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Running Neural Segmentation...</p>
            </div>
          )}

          {result && !loading && (
            <div className="fade-in">
              {activeTab === 'dashboard' ? (
                <div className="results-grid">
                  <div className="card" onClick={() => setZoomImage({ src: result.original_image, alt: 'Original Slice' })}>
                    <div className="card-header"><Brain size={14} /> Original Slice</div>
                    <div className="card-body">
                      <img src={result.original_image} alt="Original" />
                    </div>
                  </div>
                  
                  <div className="card" onClick={() => setZoomImage({ src: result.mask_image, alt: 'Prediction Mask' })}>
                    <div className="card-header"><ShieldAlert size={14} /> Prediction Mask</div>
                    <div className="card-body">
                      <img src={result.mask_image} alt="Mask" />
                    </div>
                  </div>
                  
                  <div className="card" onClick={() => setZoomImage({ src: result.crop_image, alt: 'ROI Crop' })}>
                    <div className="card-header"><Maximize size={14} /> ROI Crop</div>
                    <div className="card-body">
                      <img src={result.crop_image} alt="Crop" />
                    </div>
                  </div>
                  
                  <div className="card" onClick={() => setZoomImage({ src: result.overlay_image, alt: 'Segmented Overlay' })}>
                    <div className="card-header"><Sparkles size={14} /> Segmented Overlay</div>
                    <div className="card-body">
                      <img src={result.overlay_image} alt="Overlay" />
                    </div>
                  </div>
                  
                  <div className="card" onClick={() => setZoomImage({ src: result.grad_cam_image, alt: 'Grad-CAM Explainability' })}>
                    <div className="card-header"><Target size={14} /> Grad-CAM Explainability</div>
                    <div className="card-body">
                      <img src={result.grad_cam_image} alt="Grad-CAM" />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="results-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                   <div className="card">
                    <div className="card-header">Input MRI Scan</div>
                    <div className="card-body">
                      <img src={result.original_image} alt="Original" />
                    </div>
                  </div>
                   <div className="card" style={{ borderColor: 'var(--primary)' }}>
                    <div className="card-header" style={{ color: 'var(--primary)' }}>Analysis Overlay</div>
                    <div className="card-body">
                      <img src={result.overlay_image} alt="Overlay" />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {zoomImage && (
            <div className="zoom-modal" onClick={() => setZoomImage(null)}>
              <div className="zoom-content">
                <img src={zoomImage.src} alt={zoomImage.alt} />
                <div className="zoom-label">{zoomImage.alt}</div>
                <button className="close-zoom" onClick={() => setZoomImage(null)}>&times;</button>
              </div>
            </div>
          )}
        </section>
      </main>
      )}
    </div>
  );
}

export default App;
