import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Shield, ShieldAlert, ShieldCheck, AlertTriangle, FileText,
  UploadCloud, RefreshCw, BarChart3, Layers, Search, Eye,
  CheckCircle2, XCircle, ChevronRight, Info, ExternalLink, Cpu,
  Sparkles, Image as ImageIcon, ZoomIn, FileSearch, Activity,
  X, ChevronDown, ArrowRight, Lock, Fingerprint, ScanLine,
  AlertCircle, Monitor, Database, Settings
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || '';

const DOC_TYPE_NAMES = { aadhaar: 'Aadhaar', pan: 'PAN', dl: 'Driving License' };

const STATUS_CLASSES = {
  CLEAN: 'status-clean',
  WEAK: 'status-weak',
  MODERATE: 'status-moderate',
  STRONG: 'status-strong',
};

const DETECTOR_META = {
  ocr: { name: 'OCR Text Extraction', icon: FileText, desc: 'EasyOCR-based text recognition' },
  ela: { name: 'Error Level Analysis', icon: Layers, desc: 'Compression anomaly detection' },
  copy_move: { name: 'Copy-Move Detection', icon: ScanLine, desc: 'Cloned region detection via ORB+RANSAC' },
  typography: { name: 'Typography Analysis', icon: Fingerprint, desc: 'Stroke sharpness & font consistency' },
  metadata: { name: 'Metadata / EXIF', icon: Database, desc: 'File metadata & editing software traces' },
  field_validation: { name: 'Field Validation', icon: Lock, desc: 'Checksum, format & date verification' },
};

// ─── Helper: Verdict Badge Config ───
function getVerdictConfig(verdict) {
  switch (verdict) {
    case 'AUTHENTIC':
      return { cls: 'verdict-authentic', icon: ShieldCheck, label: 'AUTHENTIC', color: '#34d399' };
    case 'SUSPICIOUS':
      return { cls: 'verdict-suspicious', icon: AlertTriangle, label: 'SUSPICIOUS', color: '#fbbf24' };
    default:
      return { cls: 'verdict-tampered', icon: ShieldAlert, label: 'FLAGGED / TAMPERED', color: '#fb7185' };
  }
}

// ─── Detector Card Component ───
function DetectorCard({ detKey, data }) {
  const meta = DETECTOR_META[detKey] || { name: detKey, icon: Info, desc: '' };
  const IconComp = meta.icon;
  const status = data?.status || 'CLEAN';
  const confidence = data?.confidence || 0;
  const explanation = data?.explanation || '';

  return (
    <div className="detector-card animate-fadeIn">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <IconComp style={{ width: 16, height: 16, color: 'var(--text-muted)', flexShrink: 0 }} />
          <span style={{ fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-primary)' }}>{meta.name}</span>
        </div>
        <span className={`detector-status ${STATUS_CLASSES[status] || 'status-clean'}`}>{status}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
        <div className="progress-bar" style={{ flex: 1 }}>
          <div className="progress-fill" style={{
            width: `${Math.round(confidence * 100)}%`,
            background: status === 'CLEAN' ? '#10b981' : status === 'WEAK' ? '#f59e0b' : status === 'MODERATE' ? '#fb923c' : '#f43f5e',
          }} />
        </div>
        <span className="font-mono" style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', minWidth: 36, textAlign: 'right' }}>
          {Math.round(confidence * 100)}%
        </span>
      </div>
      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5 }} className="line-clamp-2">{explanation}</p>
    </div>
  );
}

// ─── Score Ring Component ───
function ScoreRing({ score, color, size = 88 }) {
  const pct = Math.max(0, Math.min(100, score));
  return (
    <div className="score-ring" style={{ width: size, height: size, '--ring-color': color, '--ring-pct': pct }}>
      <span className="font-mono" style={{ fontSize: size * 0.28, fontWeight: 900, color }}>{pct.toFixed(1)}%</span>
      <span style={{ fontSize: size * 0.1, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Score</span>
    </div>
  );
}

// ─── Main App ───
export default function App() {
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [screeningResult, setScreeningResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [error, setError] = useState(null);
  const [activeView, setActiveView] = useState('annotated');
  const [showBenchmark, setShowBenchmark] = useState(false);
  const [benchmarkMode, setBenchmarkMode] = useState('batch');
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [batchBenchmarkData, setBatchBenchmarkData] = useState(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [backendHealth, setBackendHealth] = useState('checking');
  const fileInputRef = useRef(null);

  // Health check
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/samples`);
        setBackendHealth(res.ok ? 'online' : 'offline');
        if (res.ok) {
          const data = await res.json();
          setSamples(data.samples || []);
        }
      } catch { setBackendHealth('offline'); }
    };
    check();
    const iv = setInterval(check, 30000);
    return () => clearInterval(iv);
  }, []);

  const handleSelectSample = useCallback(async (sample) => {
    setSelectedSample(sample);
    setLoading(true);
    setLoadingStatus('Initializing screening pipeline...');
    setError(null);
    setScreeningResult(null);
    try {
      const formData = new FormData();
      formData.append('sample_id', sample.id);
      setLoadingStatus('Running OCR & forensic analysis...');
      const res = await fetch(`${API_BASE}/api/screen`, { method: 'POST', body: formData });
      const result = await res.json();
      if (!res.ok) throw new Error(result.message || result.detail || 'Screening failed');
      setScreeningResult(result);
      setActiveView('annotated');
    } catch (err) { setError(err.message); } finally { setLoading(false); setLoadingStatus(''); }
  }, []);

  const handleFileUpload = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) { setError('File exceeds 10MB limit.'); return; }
    setSelectedSample({ id: 'custom_upload', filename: file.name, label: 'USER_UPLOAD' });
    setLoading(true);
    setLoadingStatus('Uploading document...');
    setError(null);
    setScreeningResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      setLoadingStatus('Running OCR & forensic analysis...');
      const res = await fetch(`${API_BASE}/api/screen`, { method: 'POST', body: formData });
      const result = await res.json();
      if (!res.ok) throw new Error(result.message || result.detail || 'Screening failed');
      setScreeningResult(result);
      setActiveView('annotated');
    } catch (err) { setError(err.message); } finally { setLoading(false); setLoadingStatus(''); }
  }, []);

  const runBatchBenchmark = useCallback(async (forceRerun = false) => {
    setShowBenchmark(true);
    setBenchmarkMode('batch');
    setBenchmarkLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark/batch${forceRerun ? '?rerun=true' : ''}`);
      if (!res.ok) throw new Error('Batch benchmark failed');
      setBatchBenchmarkData(await res.json());
    } catch (err) { console.error(err); } finally { setBenchmarkLoading(false); }
  }, []);

  const runStandardBenchmark = useCallback(async () => {
    setShowBenchmark(true);
    setBenchmarkMode('standard');
    setBenchmarkLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark?real_ocr=true`);
      if (!res.ok) throw new Error('Benchmark failed');
      setBenchmarkData(await res.json());
    } catch (err) { console.error(err); } finally { setBenchmarkLoading(false); }
  }, []);

  const r = screeningResult;
  const why = r?.why_this_verdict;
  const detExps = why?.detector_explanations || {};
  const vConf = r ? getVerdictConfig(r.verdict) : null;
  const VerdictIcon = vConf?.icon;

  // Canvas image source
  const canvasImg = useMemo(() => {
    if (!r) return null;
    switch (activeView) {
      case 'ela': return r.visualizations?.ela_heatmap;
      case 'edge': return r.visualizations?.edge_gradient_map;
      case 'copymove': return r.visualizations?.copy_move_matches;
      default: return r.original_image_b64 || r.visualizations?.ela_heatmap;
    }
  }, [r, activeView]);

  return (
    <div style={{ minHeight: '100vh' }}>
      {/* ═══ Navbar ═══ */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="brand-icon"><Shield style={{ width: 16, height: 16, color: 'white' }} /></div>
          <span>DocuShield AI</span>
          <span style={{ fontSize: '0.6875rem', fontWeight: 500, color: 'var(--text-muted)', marginLeft: 4 }}>SIH 2026</span>
        </div>
        <div className="navbar-actions">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 8 }}>
            <div className={`health-dot ${backendHealth === 'online' ? 'online' : 'offline'}`} />
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
              {backendHealth === 'online' ? 'API Connected' : backendHealth === 'checking' ? 'Connecting...' : 'API Offline'}
            </span>
          </div>
          <button className="btn btn-sm" onClick={() => runBatchBenchmark(false)}>
            <BarChart3 style={{ width: 14, height: 14 }} /> Batch Test
          </button>
          <button className="btn btn-sm" onClick={runStandardBenchmark}>
            <Activity style={{ width: 14, height: 14 }} /> Benchmark
          </button>
        </div>
      </nav>

      {/* ═══ Error Banner ═══ */}
      {error && (
        <div style={{ background: 'rgba(244, 63, 94, 0.08)', borderBottom: '1px solid rgba(244, 63, 94, 0.2)', padding: '0.625rem 1.5rem', display: 'flex', alignItems: 'center', gap: 8 }}>
          <AlertCircle style={{ width: 16, height: 16, color: '#fb7185', flexShrink: 0 }} />
          <span style={{ fontSize: '0.8125rem', color: '#fda4af', flex: 1 }}>{error}</span>
          <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: '#fb7185', cursor: 'pointer' }}>
            <X style={{ width: 14, height: 14 }} />
          </button>
        </div>
      )}

      {/* ═══ Upload Bar ═══ */}
      <div style={{ padding: '0.75rem 1.5rem', borderBottom: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', gap: 12, background: 'rgba(6, 10, 19, 0.4)' }}>
        <button className="btn btn-primary btn-sm" onClick={() => fileInputRef.current?.click()}>
          <UploadCloud style={{ width: 14, height: 14 }} /> Upload Document
        </button>
        <input ref={fileInputRef} type="file" accept=".jpg,.jpeg,.png,.webp" style={{ display: 'none' }} onChange={handleFileUpload} />
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>or select a test sample below</span>
        {loading && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
            <RefreshCw style={{ width: 14, height: 14, color: 'var(--accent-cyan)', animation: 'spin 1s linear infinite' }} />
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>{loadingStatus}</span>
          </div>
        )}
      </div>

      {/* ═══ Main Content ═══ */}
      {!r && !loading ? (
        /* ─── Landing: Sample Gallery ─── */
        <div style={{ padding: '2rem 1.5rem', maxWidth: 1200, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <h1 style={{ fontSize: '2rem', fontWeight: 900, letterSpacing: '-0.03em', marginBottom: '0.5rem' }}>
              <span style={{ background: 'linear-gradient(135deg, #06b6d4, #6366f1)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Document Forensic Screening
              </span>
            </h1>
            <p style={{ color: 'var(--text-muted)', maxWidth: 560, margin: '0 auto', fontSize: '0.9375rem' }}>
              AI-powered multimodal identity document authentication system for Aadhaar, PAN, and Driving Licence.
            </p>
          </div>

          {/* Upload Zone */}
          <div className="upload-zone" onClick={() => fileInputRef.current?.click()} style={{ maxWidth: 480, margin: '0 auto 2rem' }}>
            <UploadCloud style={{ width: 36, height: 36, color: 'var(--text-muted)', margin: '0 auto 0.75rem' }} />
            <p style={{ fontWeight: 600, marginBottom: 4, color: 'var(--text-secondary)' }}>Drop your document here or click to browse</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Supports JPEG, PNG, WebP up to 10MB</p>
          </div>

          {/* Test Samples Grid */}
          {samples.length > 0 && (
            <>
              <h2 style={{ fontSize: '0.8125rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                Synthetic Test Dataset ({samples.length} documents)
              </h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '0.75rem' }}>
                {samples.map(s => (
                  <div key={s.id}
                    className="detector-card"
                    style={{ cursor: 'pointer', transition: 'all 0.15s' }}
                    onClick={() => handleSelectSample(s)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                      <span className={`detector-status ${s.label === 'GENUINE' ? 'status-clean' : 'status-strong'}`}>
                        {s.label === 'GENUINE' ? '✓ GENUINE' : '⚠ TAMPERED'}
                      </span>
                      <span style={{ fontSize: '0.625rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--accent-cyan)', letterSpacing: '0.06em' }}>
                        {DOC_TYPE_NAMES[s.doc_type] || s.doc_type}
                      </span>
                    </div>
                    {s.thumbnail_b64 && (
                      <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', marginBottom: 6, aspectRatio: '16/10', background: 'var(--bg-surface)' }}>
                        <img src={s.thumbnail_b64} alt={s.filename} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      </div>
                    )}
                    <p className="truncate" style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}>{s.filename}</p>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      ) : (
        /* ─── Results: Two-Column Layout ─── */
        <div className="main-grid animate-fadeIn">
          {/* ═══ LEFT: Forensic Canvas ═══ */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {/* Canvas */}
            <div className="canvas-container">
              <div className="canvas-tabs">
                {[
                  { key: 'annotated', label: 'Annotated' },
                  { key: 'ela', label: 'ELA Heatmap' },
                  { key: 'edge', label: 'Edge Map' },
                  { key: 'copymove', label: 'Copy-Move' },
                  { key: 'raw', label: 'Raw Image' },
                ].map(tab => (
                  <button key={tab.key} className={`canvas-tab ${activeView === tab.key ? 'active' : ''}`} onClick={() => setActiveView(tab.key)}>
                    {tab.label}
                  </button>
                ))}
              </div>
              <div style={{ position: 'relative', background: 'var(--bg-surface)', minHeight: 300 }}>
                {loading && (
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', zIndex: 5, background: 'rgba(6, 10, 19, 0.85)' }}>
                    <div className="scanline-effect" />
                    <RefreshCw style={{ width: 32, height: 32, color: 'var(--accent-cyan)', animation: 'spin 1s linear infinite', marginBottom: 12 }} />
                    <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>{loadingStatus || 'Analyzing...'}</span>
                  </div>
                )}
                {canvasImg ? (
                  <div style={{ position: 'relative' }}>
                    <img src={canvasImg} alt="Forensic View" style={{ width: '100%', display: 'block' }} />
                    {/* Bounding box overlays for annotated view */}
                    {activeView === 'annotated' && r?.flagged_regions?.map((reg, i) => {
                      const [bx, by, bw, bh] = reg.box;
                      const iw = r.image_dimensions?.width || 1;
                      const ih = r.image_dimensions?.height || 1;
                      return (
                        <div key={i} title={`${reg.label}: ${reg.reason}`} style={{
                          position: 'absolute',
                          left: `${(bx / iw) * 100}%`, top: `${(by / ih) * 100}%`,
                          width: `${(bw / iw) * 100}%`, height: `${(bh / ih) * 100}%`,
                          border: `2px solid ${reg.color || '#ef4444'}`,
                          borderRadius: 4,
                          background: `${reg.color || '#ef4444'}15`,
                          cursor: 'pointer',
                          transition: 'all 0.15s',
                        }} />
                      );
                    })}
                  </div>
                ) : !loading ? (
                  <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                    <ImageIcon style={{ width: 48, height: 48 }} />
                  </div>
                ) : null}
              </div>
            </div>

            {/* Flagged Regions Table */}
            {r?.flagged_regions?.length > 0 && (
              <div className="panel">
                <div className="panel-header">
                  <AlertTriangle style={{ width: 14, height: 14, color: '#fbbf24' }} />
                  Flagged Regions ({r.flagged_regions.length})
                </div>
                <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                  <table className="data-table">
                    <thead><tr><th>Layer</th><th>Label</th><th>Confidence</th></tr></thead>
                    <tbody>
                      {r.flagged_regions.map((reg, i) => (
                        <tr key={i}>
                          <td style={{ color: reg.color }}>{reg.layer}</td>
                          <td>{reg.label}</td>
                          <td className="font-mono">{(reg.score * 100).toFixed(0)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* OCR Text Extraction Panel */}
            {r?.signals?.nlp_validation?.extracted_full_text && (
              <div className="panel">
                <div className="panel-header"><FileText style={{ width: 14, height: 14 }} /> Extracted OCR Text</div>
                <div className="panel-body">
                  <pre className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', lineHeight: 1.6, maxHeight: 200, overflowY: 'auto' }}>
                    {r.signals.nlp_validation.extracted_full_text}
                  </pre>
                </div>
              </div>
            )}

            {/* Back button */}
            <button className="btn" onClick={() => { setScreeningResult(null); setSelectedSample(null); }} style={{ alignSelf: 'flex-start' }}>
              <ArrowRight style={{ width: 14, height: 14, transform: 'rotate(180deg)' }} /> Back to Gallery
            </button>
          </div>

          {/* ═══ RIGHT: Results Panel ═══ */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {r && !loading && (
              <>
                {/* Verdict Banner */}
                <div className={`verdict-badge ${vConf.cls} animate-slideUp`}>
                  <VerdictIcon style={{ width: 24, height: 24 }} />
                  <div style={{ flex: 1 }}>
                    <div>{vConf.label}</div>
                    <div style={{ fontSize: '0.6875rem', fontWeight: 500, opacity: 0.8, textTransform: 'none', letterSpacing: 0 }}>
                      {r.summary_explanation}
                    </div>
                  </div>
                </div>

                {/* Score Cards Row */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem' }}>
                  <div className="stat-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <ScoreRing score={r.authenticity_score} color={vConf.color} size={72} />
                    <div className="stat-label" style={{ marginTop: 6 }}>Authenticity</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value font-mono" style={{ color: r.risk_score > 40 ? '#fb7185' : r.risk_score > 15 ? '#fbbf24' : '#34d399' }}>
                      {r.risk_score}%
                    </div>
                    <div className="stat-label">Risk Score</div>
                    <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                      <span className={`detector-status ${why?.human_review_required ? 'status-moderate' : 'status-clean'}`}>
                        {why?.human_review_required ? '⚑ REVIEW' : '✓ AUTO'}
                      </span>
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value font-mono" style={{ color: 'var(--accent-cyan)', fontSize: '1rem' }}>
                      {DOC_TYPE_NAMES[r.signals?.nlp_validation?.document_type] || 'Unknown'}
                    </div>
                    <div className="stat-label">Document Type</div>
                    <div style={{ marginTop: 4, fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
                      {r.ocr_extraction?.token_count || 0} tokens • {r.ocr_extraction?.line_count || 0} lines
                    </div>
                  </div>
                </div>

                {/* WHY THIS VERDICT */}
                <div className="panel">
                  <div className="panel-header"><Info style={{ width: 14, height: 14, color: 'var(--accent-cyan)' }} /> Why This Verdict?</div>
                  <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {/* Positive Checks */}
                    {why?.positive_checks?.length > 0 && (
                      <div>
                        <div style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#34d399', marginBottom: 6 }}>
                          ✓ Positive Verifications
                        </div>
                        {why.positive_checks.map((c, i) => (
                          <div key={i} className="evidence-item">
                            <CheckCircle2 className="evidence-icon" style={{ color: '#34d399' }} />
                            <span>{c}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Cautions */}
                    {why?.cautions?.length > 0 && (
                      <div>
                        <div style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#fbbf24', marginBottom: 6 }}>
                          ⚠ Quality Cautions
                        </div>
                        {why.cautions.map((c, i) => (
                          <div key={i} className="evidence-item">
                            <AlertTriangle className="evidence-icon" style={{ color: '#fbbf24' }} />
                            <span>{c}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Critical Evidence */}
                    {why?.critical_evidence?.length > 0 && (
                      <div>
                        <div style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#fb7185', marginBottom: 6 }}>
                          ✕ Critical / Deterministic Evidence
                        </div>
                        {why.critical_evidence.map((c, i) => (
                          <div key={i} className="evidence-item">
                            <XCircle className="evidence-icon" style={{ color: '#fb7185' }} />
                            <span style={{ color: '#fda4af' }}>{c}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* 6 Detector Cards */}
                <div>
                  <div style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                    Detector Analysis
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.625rem' }}>
                    {Object.keys(DETECTOR_META).map(key => (
                      <DetectorCard key={key} detKey={key} data={detExps[key]} />
                    ))}
                  </div>
                </div>

                {/* Field Validation Detail */}
                {r.signals?.nlp_validation?.field_checks?.length > 0 && (
                  <div className="panel">
                    <div className="panel-header"><Lock style={{ width: 14, height: 14 }} /> Field Validation Results</div>
                    <div style={{ maxHeight: 220, overflowY: 'auto' }}>
                      <table className="data-table">
                        <thead><tr><th>Field</th><th>Value</th><th>Status</th></tr></thead>
                        <tbody>
                          {r.signals.nlp_validation.field_checks.map((f, i) => (
                            <tr key={i}>
                              <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{f.field}</td>
                              <td className="font-mono truncate" style={{ maxWidth: 180 }}>{f.value || '—'}</td>
                              <td>
                                <span className={`detector-status ${f.status === 'PASS' ? 'status-clean' : f.status === 'FAIL' ? 'status-strong' : 'status-weak'}`}>
                                  {f.status}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Condition Assessment */}
                {r.condition_assessment && (
                  <div className="panel">
                    <div className="panel-header"><Monitor style={{ width: 14, height: 14 }} /> Image Condition</div>
                    <div className="panel-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.5rem', fontSize: '0.75rem' }}>
                      <div>
                        <div style={{ color: 'var(--text-muted)', fontWeight: 600, marginBottom: 2 }}>Resolution</div>
                        <div className="font-mono" style={{ color: 'var(--text-primary)' }}>
                          {r.condition_assessment.resolution?.width}×{r.condition_assessment.resolution?.height}
                        </div>
                      </div>
                      <div>
                        <div style={{ color: 'var(--text-muted)', fontWeight: 600, marginBottom: 2 }}>Blur</div>
                        <div className="font-mono" style={{ color: r.condition_assessment.is_blurry ? '#fbbf24' : '#34d399' }}>
                          {r.condition_assessment.blur_level || 'N/A'}
                        </div>
                      </div>
                      <div>
                        <div style={{ color: 'var(--text-muted)', fontWeight: 600, marginBottom: 2 }}>Compression</div>
                        <div className="font-mono" style={{ color: r.condition_assessment.is_heavily_compressed ? '#fbbf24' : '#34d399' }}>
                          {r.condition_assessment.compression_level || 'N/A'}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Fusion Contributions */}
                {r.fusion_contributions && (
                  <div className="panel">
                    <div className="panel-header"><Layers style={{ width: 14, height: 14 }} /> Fusion Contributions</div>
                    <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {['nlp', 'ela', 'font', 'copy_move', 'metadata'].map(k => {
                        const fc = r.fusion_contributions[k];
                        if (!fc) return null;
                        const label = k === 'nlp' ? 'NLP / Fields' : k === 'ela' ? 'ELA' : k === 'font' ? 'Typography' : k === 'copy_move' ? 'Copy-Move' : 'Metadata';
                        return (
                          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', minWidth: 80 }}>{label}</span>
                            <div className="progress-bar" style={{ flex: 1 }}>
                              <div className="progress-fill" style={{
                                width: `${fc.raw_score}%`,
                                background: fc.raw_score >= 90 ? '#10b981' : fc.raw_score >= 70 ? '#f59e0b' : '#f43f5e',
                              }} />
                            </div>
                            <span className="font-mono" style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', minWidth: 36, textAlign: 'right' }}>{fc.raw_score}%</span>
                            <span style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>×{fc.weight}</span>
                          </div>
                        );
                      })}
                      <div style={{ marginTop: 4, fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        {r.fusion_contributions.penalties_applied?.map((p, i) => <div key={i}>→ {p}</div>)}
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* ═══ Benchmark Modal ═══ */}
      {showBenchmark && (
        <div className="modal-backdrop" onClick={(e) => { if (e.target === e.currentTarget) setShowBenchmark(false); }}>
          <div className="modal-content animate-slideUp">
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <BarChart3 style={{ width: 20, height: 20, color: 'var(--accent-cyan)' }} />
                <h2 style={{ fontSize: '1.125rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
                  {benchmarkMode === 'batch' ? 'Real-World Batch Benchmark' : 'Synthetic Benchmark'}
                </h2>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className={`btn btn-sm ${benchmarkMode === 'batch' ? 'btn-primary' : ''}`} onClick={() => { setBenchmarkMode('batch'); if (!batchBenchmarkData) runBatchBenchmark(false); }}>
                  Batch (D:\)
                </button>
                <button className={`btn btn-sm ${benchmarkMode === 'standard' ? 'btn-primary' : ''}`} onClick={() => { setBenchmarkMode('standard'); if (!benchmarkData) runStandardBenchmark(); }}>
                  Synthetic
                </button>
                <button className="btn btn-sm btn-icon" onClick={() => setShowBenchmark(false)}>
                  <X style={{ width: 16, height: 16 }} />
                </button>
              </div>
            </div>
            <div className="modal-body">
              {benchmarkLoading ? (
                <div style={{ textAlign: 'center', padding: '3rem 0' }}>
                  <RefreshCw style={{ width: 32, height: 32, color: 'var(--accent-cyan)', animation: 'spin 1s linear infinite', margin: '0 auto 1rem' }} />
                  <p style={{ color: 'var(--text-muted)' }}>Running benchmark pipeline...</p>
                </div>
              ) : benchmarkMode === 'batch' && batchBenchmarkData ? (
                <BenchmarkResults data={batchBenchmarkData} onRerun={() => runBatchBenchmark(true)} />
              ) : benchmarkMode === 'standard' && benchmarkData ? (
                <BenchmarkResults data={benchmarkData} onRerun={runStandardBenchmark} />
              ) : (
                <div style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--text-muted)' }}>
                  <p>No benchmark data available. Click a benchmark mode above to run.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Benchmark Results Component ───
function BenchmarkResults({ data, onRerun }) {
  const m = data?.metrics || {};
  const hasFailures = (data?.failures?.length || 0) > 0;
  const allRecords = data?.all_records || data?.details || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Action bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
          {data?.total_documents || allRecords.length} documents evaluated
          {data?.benchmark_timestamp ? ` • ${new Date(data.benchmark_timestamp).toLocaleString()}` : ''}
        </span>
        <button className="btn btn-sm" onClick={onRerun}>
          <RefreshCw style={{ width: 14, height: 14 }} /> Re-run
        </button>
      </div>

      {/* Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem' }}>
        <div className="stat-card"><div className="stat-value" style={{ color: '#34d399' }}>{m.accuracy ?? m.accuracy_pct ?? '—'}%</div><div className="stat-label">Accuracy</div></div>
        <div className="stat-card"><div className="stat-value" style={{ color: '#22d3ee' }}>{m.precision ?? '—'}%</div><div className="stat-label">Precision</div></div>
        <div className="stat-card"><div className="stat-value" style={{ color: '#818cf8' }}>{m.recall ?? '—'}%</div><div className="stat-label">Recall</div></div>
        <div className="stat-card"><div className="stat-value" style={{ color: '#fbbf24' }}>{m.f1_score ?? m.f1 ?? '—'}%</div><div className="stat-label">F1 Score</div></div>
      </div>

      {/* Confusion Matrix */}
      <div>
        <div style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 8 }}>
          Confusion Matrix
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.625rem', maxWidth: 400 }}>
          <div className="cm-cell cm-tp">
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{m.tp ?? m.true_positives ?? 0}</div>
            <div style={{ fontSize: '0.6875rem' }}>True Positive</div>
          </div>
          <div className="cm-cell cm-fp">
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{m.fp ?? m.false_positives ?? 0}</div>
            <div style={{ fontSize: '0.6875rem' }}>False Positive</div>
          </div>
          <div className="cm-cell cm-fn">
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{m.fn ?? m.false_negatives ?? 0}</div>
            <div style={{ fontSize: '0.6875rem' }}>False Negative</div>
          </div>
          <div className="cm-cell cm-tn">
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{m.tn ?? m.true_negatives ?? 0}</div>
            <div style={{ fontSize: '0.6875rem' }}>True Negative</div>
          </div>
        </div>
      </div>

      {/* Per-Document Results Table */}
      {allRecords.length > 0 && (
        <div>
          <div style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 8 }}>
            Per-Document Results
          </div>
          <div style={{ maxHeight: 300, overflowY: 'auto', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-subtle)' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Expected</th>
                  <th>Verdict</th>
                  <th>Score</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {allRecords.map((item, idx) => {
                  const isCorrect = (item.status_tag || item.status || '').includes('CORRECT');
                  return (
                    <tr key={idx}>
                      <td style={{ fontWeight: 600, color: 'var(--text-primary)', maxWidth: 200 }} className="truncate">{item.filename}</td>
                      <td>{item.expected_label || item.actual_label || '—'}</td>
                      <td>
                        <span style={{ color: (item.final_verdict || item.predicted_verdict) === 'AUTHENTIC' ? '#34d399' : (item.final_verdict || item.predicted_verdict) === 'SUSPICIOUS' ? '#fbbf24' : '#fb7185' }}>
                          {item.final_verdict || item.predicted_verdict}
                        </span>
                      </td>
                      <td className="font-mono" style={{ fontWeight: 700, color: 'white' }}>{item.authenticity_score}%</td>
                      <td>
                        <span className={`detector-status ${isCorrect ? 'status-clean' : 'status-strong'}`}>
                          {isCorrect ? '✓ Correct' : '✕ Wrong'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Failures Detail */}
      {hasFailures && (
        <div>
          <div style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#fb7185', marginBottom: 8 }}>
            Failure Analysis ({data.failures.length})
          </div>
          {data.failures.map((f, i) => (
            <div key={i} style={{ padding: '0.75rem', background: 'rgba(244, 63, 94, 0.04)', border: '1px solid rgba(244, 63, 94, 0.1)', borderRadius: 'var(--radius-md)', marginBottom: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontWeight: 700, color: '#fda4af', fontSize: '0.8125rem' }}>{f.image_name}</span>
                <span className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{f.authenticity_score}%</span>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Expected: {f.expected_result} → Got: {f.final_verdict} | Category: {f.failure_category}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
