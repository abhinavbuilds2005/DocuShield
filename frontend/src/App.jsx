import React, { useState, useEffect, useRef } from 'react';
import {
  Shield, ShieldAlert, ShieldCheck, AlertTriangle, FileText,
  UploadCloud, RefreshCw, BarChart3, Layers, Search, Eye,
  CheckCircle2, XCircle, ChevronRight, Info, ExternalLink, Cpu,
  Sparkles, Sliders, Image as ImageIcon, ZoomIn, FileSearch
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || '';

export default function App() {
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [screeningResult, setScreeningResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [error, setError] = useState(null);
  
  // View Controls
  const [activeView, setActiveView] = useState('annotated'); // 'annotated', 'ela', 'edge', 'raw'
  const [hoveredBox, setHoveredBox] = useState(null);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'nlp', 'forensics', 'fields'
  
  // Benchmark Modal
  const [showBenchmark, setShowBenchmark] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);

  const fileInputRef = useRef(null);

  // Load sample documents on mount
  useEffect(() => {
    fetchSamples();
  }, []);

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (!res.ok) throw new Error('Could not fetch mock dataset catalog');
      const data = await res.json();
      setSamples(data.samples || []);
      // Pre-select first genuine Aadhaar by default
      if (data.samples?.length > 0) {
        handleSelectSample(data.samples[0]);
      }
    } catch (err) {
      console.error(err);
      setError('Backend connection error. Please ensure FastAPI server is running on port 8008.');
    }
  };

  const handleSelectSample = async (sample) => {
    setSelectedSample(sample);
    setLoading(true);
    setLoadingStatus('Sending to screening pipeline...');
    setError(null);
    try {
      const formData = new FormData();
      formData.append('sample_id', sample.id);
      setLoadingStatus('Running OCR & forensic analysis...');
      const res = await fetch(`${API_BASE}/api/screen`, {
        method: 'POST',
        body: formData,
      });
      const result = await res.json();
      if (!res.ok) {
        throw new Error(result.message || result.detail || 'Screening failed on sample document');
      }
      setScreeningResult(result);
      setActiveView('annotated');
      setLoadingStatus('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStatus('');
    }
  };

  const MAX_UPLOAD_MB = 10;
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Client-side validation
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`File too large (${(file.size / (1024*1024)).toFixed(1)} MB). Maximum is ${MAX_UPLOAD_MB} MB.`);
      return;
    }
    if (!ALLOWED_TYPES.includes(file.type) && !file.name.match(/\.(jpe?g|png|webp)$/i)) {
      setError('Unsupported format. Please upload a JPEG, PNG, or WebP image.');
      return;
    }

    setSelectedSample({ id: 'custom_upload', filename: file.name, label: 'USER_UPLOAD' });
    setLoading(true);
    setLoadingStatus('Uploading document...');
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      setLoadingStatus('Running OCR extraction...');
      const res = await fetch(`${API_BASE}/api/screen`, {
        method: 'POST',
        body: formData,
      });
      setLoadingStatus('Processing forensic analysis...');
      const result = await res.json();
      if (!res.ok) {
        throw new Error(result.message || result.detail || 'Screening analysis failed on uploaded document');
      }
      setScreeningResult(result);
      setActiveView('annotated');
      setLoadingStatus('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStatus('');
    }
  };

  const runBenchmarkEvaluation = async () => {
    setShowBenchmark(true);
    setBenchmarkLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark`);
      if (!res.ok) throw new Error('Benchmark failed to execute');
      const data = await res.json();
      setBenchmarkData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setBenchmarkLoading(false);
    }
  };

  const getVerdictBadge = (verdict) => {
    switch (verdict) {
      case 'AUTHENTIC':
        return {
          bg: 'bg-emerald-500/10 border-emerald-500/40 text-emerald-400',
          glow: 'glow-emerald',
          icon: <ShieldCheck className="w-6 h-6 text-emerald-400" />,
          label: 'AUTHENTIC DOCUMENT'
        };
      case 'SUSPICIOUS':
        return {
          bg: 'bg-amber-500/10 border-amber-500/40 text-amber-400',
          glow: 'glow-amber',
          icon: <AlertTriangle className="w-6 h-6 text-amber-400" />,
          label: 'SUSPICIOUS / REVIEW NEEDED'
        };
      default:
        return {
          bg: 'bg-rose-500/10 border-rose-500/40 text-rose-400',
          glow: 'glow-rose',
          icon: <ShieldAlert className="w-6 h-6 text-rose-400" />,
          label: 'FLAGGED / TAMPERED'
        };
    }
  };

  const verdictBadge = screeningResult ? getVerdictBadge(screeningResult.verdict) : null;

  return (
    <div className="min-h-screen pb-16">
      {/* Top Notification Bar & Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 text-white shadow-lg shadow-cyan-500/20">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                  DocuShield AI
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800/60 text-cyan-300 font-mono">
                  SIH26188
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                AI-Based Fake Identity & Document Screening System (NLP + Image Forensics Fusion)
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={runBenchmarkEvaluation}
              className="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/30 text-indigo-300 hover:text-indigo-200 text-xs font-semibold transition"
            >
              <BarChart3 className="w-4 h-4" />
              <span>Dataset Benchmark (20 Docs)</span>
            </button>

            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-xs font-semibold shadow-md shadow-cyan-500/25 transition"
            >
              <UploadCloud className="w-4 h-4" />
              <span>Upload Document</span>
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".jpg,.jpeg,.png,.webp"
              className="hidden"
            />
            <span className="text-xs text-slate-500 hidden sm:block">JPG / PNG / WebP · Max 10 MB</span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 space-y-6">
        
        {/* Error Alert */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-950/50 border border-rose-800/50 text-rose-300 flex items-center justify-between text-sm">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
            <button onClick={fetchSamples} className="px-3 py-1 bg-rose-900/60 rounded text-xs hover:bg-rose-800">
              Retry
            </button>
          </div>
        )}

        {/* Quick Test Sample Selector Grid */}
        <section className="glass-panel rounded-2xl p-4 sm:p-5 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
                <FileSearch className="w-4 h-4 text-cyan-400" />
                <span>Quick Test Synthetic Dataset (Pre-Loaded Baseline & Tampered Pairs)</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Click any mock card below to test the fused detection pipeline in real time (Watermarked mock identities)
              </p>
            </div>
            <span className="text-xs text-slate-400 font-mono self-start sm:self-auto bg-slate-900/90 px-2.5 py-1 rounded-md border border-slate-800">
              {samples.length} Paired Cards
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-7 gap-2.5 max-h-48 overflow-y-auto pr-1">
            {samples.map((s) => {
              const isSelected = selectedSample?.id === s.id;
              const isGenuine = s.label === 'GENUINE';
              return (
                <button
                  key={s.id}
                  onClick={() => handleSelectSample(s)}
                  className={`text-left p-2 rounded-xl border transition-all relative flex flex-col justify-between overflow-hidden group ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-400 shadow-md shadow-cyan-500/20'
                      : 'bg-slate-900/60 hover:bg-slate-800/60 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="aspect-[16/10] w-full rounded-lg overflow-hidden bg-slate-950/80 mb-2 relative border border-slate-800/40">
                    {s.thumbnail_b64 ? (
                      <img
                        src={s.thumbnail_b64}
                        alt={s.name}
                        className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-slate-600">
                        <ImageIcon className="w-6 h-6" />
                      </div>
                    )}
                    <span
                      className={`absolute top-1 right-1 text-[9px] font-bold px-1.5 py-0.5 rounded ${
                        isGenuine
                          ? 'bg-emerald-500/90 text-slate-950'
                          : 'bg-rose-500/90 text-white'
                      }`}
                    >
                      {isGenuine ? 'GENUINE' : 'TAMPERED'}
                    </span>
                  </div>

                  <div>
                    <div className="text-[11px] font-semibold text-slate-200 truncate">{s.name}</div>
                    <div className="text-[10px] text-slate-400 uppercase tracking-wider mt-0.5">
                      {s.doc_type}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        {/* Screening Results Workspace */}
        {screeningResult && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Left Column: Visual Document Canvas & Layer Toggles (7 cols) */}
            <div className="lg:col-span-7 space-y-4">
              <div className="glass-panel rounded-2xl p-4 sm:p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-base font-bold text-white flex items-center space-x-2">
                      <Eye className="w-4 h-4 text-cyan-400" />
                      <span>Document Forensic Canvas</span>
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Inspecting: <span className="font-mono text-slate-300">{screeningResult.filename}</span>
                    </p>
                  </div>

                  {/* View Mode Switcher */}
                  <div className="flex items-center space-x-1 p-1 bg-slate-950/80 rounded-xl border border-slate-800 text-xs">
                    <button
                      onClick={() => setActiveView('annotated')}
                      className={`px-3 py-1 rounded-lg transition font-medium ${
                        activeView === 'annotated'
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Annotated BBoxes ({screeningResult.flagged_regions?.length || 0})
                    </button>
                    <button
                      onClick={() => setActiveView('ela')}
                      className={`px-3 py-1 rounded-lg transition font-medium ${
                        activeView === 'ela'
                          ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30 shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      ELA Heatmap
                    </button>
                    <button
                      onClick={() => setActiveView('edge')}
                      className={`px-3 py-1 rounded-lg transition font-medium ${
                        activeView === 'edge'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Edge Discontinuity
                    </button>
                    <button
                      onClick={() => setActiveView('raw')}
                      className={`px-3 py-1 rounded-lg transition font-medium ${
                        activeView === 'raw'
                          ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Raw
                    </button>
                  </div>
                </div>

                {/* Document Display Canvas */}
                <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800/80 flex items-center justify-center">
                  {loading && (
                    <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm z-30 flex flex-col items-center justify-center space-y-3">
                      <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
                      <p className="text-xs text-cyan-300 font-mono animate-pulse">{loadingStatus || 'Running Fused Forensics Pipeline...'}</p>
                    </div>
                  )}

                  {/* Scanline Effect */}
                  <div className="scanline-effect" />

                  {/* Render based on selected View Mode */}
                  {activeView === 'annotated' && (
                    <div className="relative w-full h-full flex items-center justify-center">
                      <img
                        src={screeningResult.original_image_data_uri}
                        alt="Document View"
                        className="max-w-full max-h-full object-contain"
                      />

                      {/* Flagged Bounding Boxes SVG Overlay */}
                      <svg
                        className="absolute inset-0 w-full h-full pointer-events-auto"
                        viewBox={`0 0 ${screeningResult.image_dimensions?.width || 800} ${screeningResult.image_dimensions?.height || 500}`}
                        preserveAspectRatio="xMidYMid meet"
                      >
                        {screeningResult.flagged_regions?.map((reg, idx) => {
                          const [bx, by, bw, bh] = reg.box;
                          const isHovered = hoveredBox === idx;
                          return (
                            <g
                              key={idx}
                              onMouseEnter={() => setHoveredBox(idx)}
                              onMouseLeave={() => setHoveredBox(null)}
                              className="cursor-pointer transition duration-150"
                            >
                              <rect
                                x={bx}
                                y={by}
                                width={bw}
                                height={bh}
                                fill={reg.color ? `${reg.color}25` : 'rgba(239, 68, 68, 0.2)'}
                                stroke={reg.color || '#ef4444'}
                                strokeWidth={isHovered ? 4 : 2.5}
                                strokeDasharray={isHovered ? 'none' : '4 2'}
                                className="transition-all"
                              />
                              <rect
                                x={bx}
                                y={Math.max(0, by - 22)}
                                width={Math.min(bw, 180)}
                                height={22}
                                fill={reg.color || '#ef4444'}
                                rx={3}
                              />
                              <text
                                x={bx + 6}
                                y={Math.max(14, by - 7)}
                                fill="#ffffff"
                                fontSize="11"
                                fontWeight="bold"
                                fontFamily="sans-serif"
                              >
                                {reg.label?.slice(0, 24)}
                              </text>
                            </g>
                          );
                        })}
                      </svg>
                    </div>
                  )}

                  {activeView === 'ela' && (
                    <img
                      src={screeningResult.visualizations?.ela_heatmap || screeningResult.original_image_data_uri}
                      alt="ELA Heatmap"
                      className="max-w-full max-h-full object-contain"
                    />
                  )}

                  {activeView === 'edge' && (
                    <img
                      src={screeningResult.visualizations?.edge_gradient_map || screeningResult.original_image_data_uri}
                      alt="Edge Gradient Map"
                      className="max-w-full max-h-full object-contain"
                    />
                  )}

                  {activeView === 'raw' && (
                    <img
                      src={screeningResult.original_image_data_uri}
                      alt="Raw Document"
                      className="max-w-full max-h-full object-contain"
                    />
                  )}
                </div>

                {/* Hovered / Selected Region Inspector Card */}
                {hoveredBox !== null && screeningResult.flagged_regions?.[hoveredBox] && (
                  <div className="p-3 rounded-xl bg-slate-900/95 border border-cyan-500/40 text-xs space-y-1 animate-in fade-in zoom-in-95 duration-150">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-cyan-300">
                        {screeningResult.flagged_regions[hoveredBox].label}
                      </span>
                      <span className="font-mono text-slate-400">
                        Confidence: {(screeningResult.flagged_regions[hoveredBox].score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-slate-300 text-[11px]">
                      {screeningResult.flagged_regions[hoveredBox].reason}
                    </p>
                    <div className="text-[10px] text-slate-500">
                      Layer Attribution: {screeningResult.flagged_regions[hoveredBox].layer}
                    </div>
                  </div>
                )}

                {/* Flagged Regions Quick Summary Bar */}
                <div className="flex items-center justify-between text-xs text-slate-400 pt-1">
                  <span>
                    Detected Anomalies: <strong className="text-slate-200">{screeningResult.flagged_regions?.length || 0}</strong>
                  </span>
                  <span>
                    Resolution: <strong className="text-slate-200">{screeningResult.image_dimensions?.width}x{screeningResult.image_dimensions?.height}</strong>
                  </span>
                </div>
              </div>
            </div>

            {/* Right Column: Authenticity Scorecard, Verdict & Multimodal Signals (5 cols) */}
            <div className="lg:col-span-5 space-y-4">
              
              {/* Authenticity Gauge Card */}
              <div className={`glass-panel rounded-2xl p-5 border ${verdictBadge?.bg} ${verdictBadge?.glow} space-y-4`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {verdictBadge?.icon}
                    <div>
                      <h3 className="text-base font-extrabold tracking-tight">
                        {verdictBadge?.label}
                      </h3>
                      <p className="text-xs text-slate-400">Fused Dual-Layer Authenticity Verdict</p>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-3xl font-black font-mono tracking-tight text-white">
                      {screeningResult.authenticity_score}%
                    </div>
                    <div className="text-[10px] uppercase font-bold text-slate-400">Authenticity Score</div>
                  </div>
                </div>

                {/* Score Progress Bar */}
                <div className="w-full bg-slate-900/80 rounded-full h-2.5 overflow-hidden border border-slate-800">
                  <div
                    className={`h-full transition-all duration-700 ${
                      screeningResult.authenticity_score >= 80
                        ? 'bg-emerald-500'
                        : screeningResult.authenticity_score >= 50
                        ? 'bg-amber-500'
                        : 'bg-rose-500'
                    }`}
                    style={{ width: `${screeningResult.authenticity_score}%` }}
                  />
                </div>

                <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-800/80">
                  <span className="text-slate-400">Diagnostic Status:</span>
                  <span className={`px-2 py-0.5 rounded font-mono font-bold text-[11px] ${
                    screeningResult.diagnostic_status === 'DOCUMENT_STRONGLY_SUSPECTED_TAMPERED'
                      ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                      : screeningResult.diagnostic_status === 'ANOMALY_DETECTED'
                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                      : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                  }`}>
                    {screeningResult.diagnostic_status || 'CLEAN'}
                  </span>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed">
                  {screeningResult.summary_explanation}
                </p>

                {/* Critical Triggers Alert list */}
                {screeningResult.critical_triggers?.length > 0 && (
                  <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
                    <div className="text-[11px] font-bold uppercase tracking-wider text-rose-400 flex items-center space-x-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      <span>Critical Triggers Triggered:</span>
                    </div>
                    <ul className="space-y-1">
                      {screeningResult.critical_triggers.map((trig, idx) => (
                        <li key={idx} className="text-xs text-slate-300 flex items-start space-x-2">
                          <span className="text-rose-500 font-bold">•</span>
                          <span>{trig}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Multimodal Signal Tabs */}
              <div className="glass-panel rounded-2xl p-4 sm:p-5 space-y-4">
                <div className="flex border-b border-slate-800 pb-2 space-x-1 text-xs">
                  <button
                    onClick={() => setActiveTab('overview')}
                    className={`px-3 py-1.5 rounded-lg font-semibold transition ${
                      activeTab === 'overview'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Signals Breakdown
                  </button>
                  <button
                    onClick={() => setActiveTab('nlp')}
                    className={`px-3 py-1.5 rounded-lg font-semibold transition ${
                      activeTab === 'nlp'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    NLP / OCR Layer
                  </button>
                  <button
                    onClick={() => setActiveTab('fields')}
                    className={`px-3 py-1.5 rounded-lg font-semibold transition ${
                      activeTab === 'fields'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Field Verification
                  </button>
                </div>

                {/* Tab Content: Overview Breakdown */}
                {activeTab === 'overview' && (
                  <div className="space-y-3">
                    {/* Signal 1: NLP Validation */}
                    <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/80 flex items-center justify-between">
                      <div>
                        <div className="text-xs font-bold text-slate-200 flex items-center space-x-1.5">
                          <FileText className="w-3.5 h-3.5 text-cyan-400" />
                          <span>Text & NLP Field Validation</span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-0.5">
                          Checksums, regex formats, chronological consistency
                        </p>
                      </div>
                      <div className="text-right flex items-center space-x-2">
                        <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase ${
                          screeningResult.evidence_strengths?.nlp === 'STRONG'
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : screeningResult.evidence_strengths?.nlp === 'MODERATE'
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            : screeningResult.evidence_strengths?.nlp === 'WEAK'
                            ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {screeningResult.evidence_strengths?.nlp || 'CLEAN'}
                        </span>
                        <span className="text-sm font-bold font-mono text-white">
                          {screeningResult.signals?.nlp_validation?.score}%
                        </span>
                      </div>
                    </div>

                    {/* Signal 2: ELA Forensics */}
                    <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/80 flex items-center justify-between">
                      <div>
                        <div className="text-xs font-bold text-slate-200 flex items-center space-x-1.5">
                          <Layers className="w-3.5 h-3.5 text-rose-400" />
                          <span>Error Level Analysis (ELA)</span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-0.5">
                          JPEG recompression variance & photo hotspots
                        </p>
                      </div>
                      <div className="text-right flex items-center space-x-2">
                        <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase ${
                          screeningResult.evidence_strengths?.ela === 'STRONG'
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : screeningResult.evidence_strengths?.ela === 'MODERATE'
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            : screeningResult.evidence_strengths?.ela === 'WEAK'
                            ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {screeningResult.evidence_strengths?.ela || 'CLEAN'}
                        </span>
                        <span className="text-sm font-bold font-mono text-white">
                          {screeningResult.signals?.ela_forensics?.score}%
                        </span>
                      </div>
                    </div>

                    {/* Signal 3: Typography & Font Alignment */}
                    <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/80 flex items-center justify-between">
                      <div>
                        <div className="text-xs font-bold text-slate-200 flex items-center space-x-1.5">
                          <Cpu className="w-3.5 h-3.5 text-amber-400" />
                          <span>Typography & Font Alignment</span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-0.5">
                          Anti-aliasing consistency & edge gradients
                        </p>
                      </div>
                      <div className="text-right flex items-center space-x-2">
                        <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase ${
                          screeningResult.evidence_strengths?.font === 'STRONG'
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : screeningResult.evidence_strengths?.font === 'MODERATE'
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            : screeningResult.evidence_strengths?.font === 'WEAK'
                            ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {screeningResult.evidence_strengths?.font || 'CLEAN'}
                        </span>
                        <span className="text-sm font-bold font-mono text-white">
                          {screeningResult.signals?.font_typography?.score}%
                        </span>
                      </div>
                    </div>

                    {/* Signal 4: Copy-Move Cloned Texture */}
                    <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/80 flex items-center justify-between">
                      <div>
                        <div className="text-xs font-bold text-slate-200 flex items-center space-x-1.5">
                          <Sliders className="w-3.5 h-3.5 text-indigo-400" />
                          <span>Copy-Move Forgery Detection</span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-0.5">
                          Keypoint spatial displacement & cloned stamps
                        </p>
                      </div>
                      <div className="text-right flex items-center space-x-2">
                        <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase ${
                          screeningResult.evidence_strengths?.copy_move === 'STRONG'
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : screeningResult.evidence_strengths?.copy_move === 'MODERATE'
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            : screeningResult.evidence_strengths?.copy_move === 'WEAK'
                            ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {screeningResult.evidence_strengths?.copy_move || 'CLEAN'}
                        </span>
                        <span className="text-sm font-bold font-mono text-white">
                          {screeningResult.signals?.copy_move?.score}%
                        </span>
                      </div>
                    </div>

                    {/* Signal 5: Metadata & EXIF */}
                    <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/80 flex items-center justify-between">
                      <div>
                        <div className="text-xs font-bold text-slate-200 flex items-center space-x-1.5">
                          <Info className="w-3.5 h-3.5 text-emerald-400" />
                          <span>Metadata & EXIF Inspection</span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-0.5">
                          Editing software signatures (Photoshop, GIMP)
                        </p>
                      </div>
                      <div className="text-right flex items-center space-x-2">
                        <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase ${
                          screeningResult.evidence_strengths?.metadata === 'STRONG'
                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : screeningResult.evidence_strengths?.metadata === 'MODERATE'
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            : screeningResult.evidence_strengths?.metadata === 'WEAK'
                            ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                            : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {screeningResult.evidence_strengths?.metadata || 'CLEAN'}
                        </span>
                        <span className="text-sm font-bold font-mono text-white">
                          {screeningResult.signals?.metadata_forensics?.score}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Tab Content: NLP / OCR */}
                {activeTab === 'nlp' && (
                  <div className="space-y-3 text-xs">
                    <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
                      <div className="text-slate-400 text-[11px]">Detected Document Family:</div>
                      <div className="font-bold text-white uppercase font-mono">
                        {screeningResult.signals?.nlp_validation?.document_type || 'Unknown'}
                      </div>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1 max-h-44 overflow-y-auto font-mono text-[11px] text-slate-300 leading-relaxed">
                      <div className="text-slate-400 font-sans text-[11px] mb-1">OCR Extracted Text:</div>
                      {screeningResult.signals?.nlp_validation?.extracted_full_text || 'No text extracted.'}
                    </div>
                  </div>
                )}

                {/* Tab Content: Extracted Fields */}
                {activeTab === 'fields' && (
                  <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                    {screeningResult.signals?.nlp_validation?.field_checks?.map((field, idx) => {
                      const isPass = field.status === 'PASS';
                      return (
                        <div
                          key={idx}
                          className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-start justify-between text-xs"
                        >
                          <div className="space-y-0.5">
                            <div className="font-semibold text-slate-200">{field.field}</div>
                            <div className="font-mono text-cyan-300 text-[11px]">{field.value}</div>
                            <div className="text-[10px] text-slate-400">{field.details}</div>
                          </div>
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider shrink-0 ${
                              field.status === 'PASS'
                                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                : field.status === 'UNCERTAIN'
                                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                                : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            }`}
                          >
                            {field.status}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Benchmark Evaluation Modal */}
      {showBenchmark && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="glass-panel max-w-4xl w-full max-h-[90vh] rounded-3xl p-6 overflow-y-auto space-y-6 border border-slate-700 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-xl bg-indigo-600/30 text-indigo-400 border border-indigo-500/40">
                  <BarChart3 className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">
                    SIH 2026 Dataset Benchmark Evaluation
                  </h3>
                  <p className="text-xs text-slate-400">
                    20-Document Synthetic Baseline & Tampering Test Suite (Ground-Truth Evaluated)
                  </p>
                </div>
              </div>

              <button
                onClick={() => setShowBenchmark(false)}
                className="p-2 rounded-xl text-slate-400 hover:text-white bg-slate-900 border border-slate-800"
              >
                ✕
              </button>
            </div>

            {benchmarkLoading ? (
              <div className="py-20 flex flex-col items-center justify-center space-y-3">
                <RefreshCw className="w-10 h-10 text-indigo-400 animate-spin" />
                <p className="text-sm font-mono text-indigo-300 animate-pulse">
                  Screening all 20 documents through dual-layer fusion engine...
                </p>
              </div>
            ) : benchmarkData ? (
              <div className="space-y-6">
                {/* Metric Summary Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 text-center">
                    <div className="text-2xl font-black font-mono text-emerald-400">
                      {benchmarkData.metrics?.accuracy}%
                    </div>
                    <div className="text-xs text-slate-400 mt-1 uppercase font-semibold">Overall Accuracy</div>
                  </div>

                  <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 text-center">
                    <div className="text-2xl font-black font-mono text-cyan-400">
                      {benchmarkData.metrics?.precision}%
                    </div>
                    <div className="text-xs text-slate-400 mt-1 uppercase font-semibold">Precision</div>
                  </div>

                  <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 text-center">
                    <div className="text-2xl font-black font-mono text-indigo-400">
                      {benchmarkData.metrics?.recall}%
                    </div>
                    <div className="text-xs text-slate-400 mt-1 uppercase font-semibold">Recall</div>
                  </div>

                  <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 text-center">
                    <div className="text-2xl font-black font-mono text-amber-400">
                      {benchmarkData.metrics?.f1_score}%
                    </div>
                    <div className="text-xs text-slate-400 mt-1 uppercase font-semibold">F1 Score</div>
                  </div>
                </div>

                {/* Confusion Matrix Visual */}
                <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Confusion Matrix (20 Documents)
                  </h4>
                  <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                    <div className="p-3 rounded-xl bg-emerald-950/30 border border-emerald-800/40 text-emerald-300">
                      <div className="text-base font-bold">{benchmarkData.metrics?.true_positives}</div>
                      <div>True Positives (Tampered Correctly Flagged)</div>
                    </div>
                    <div className="p-3 rounded-xl bg-cyan-950/30 border border-cyan-800/40 text-cyan-300">
                      <div className="text-base font-bold">{benchmarkData.metrics?.true_negatives}</div>
                      <div>True Negatives (Genuine Verified Authentic)</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800 text-slate-400">
                      <div className="text-base font-bold">{benchmarkData.metrics?.false_positives}</div>
                      <div>False Positives (Genuine Incorrectly Flagged)</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800 text-slate-400">
                      <div className="text-base font-bold">{benchmarkData.metrics?.false_negatives}</div>
                      <div>False Negatives (Tampered Missed)</div>
                    </div>
                  </div>
                </div>

                {/* Detailed Table */}
                <div className="space-y-2">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Per-Document Benchmark Results
                  </h4>
                  <div className="border border-slate-800 rounded-xl overflow-hidden max-h-56 overflow-y-auto">
                    <table className="w-full text-left text-xs font-mono">
                      <thead className="bg-slate-900 text-slate-400 border-b border-slate-800 sticky top-0">
                        <tr>
                          <th className="p-2.5">Document</th>
                          <th className="p-2.5">Actual</th>
                          <th className="p-2.5">Verdict</th>
                          <th className="p-2.5">Score</th>
                          <th className="p-2.5">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {benchmarkData.details?.map((item, idx) => (
                          <tr key={idx} className="hover:bg-slate-900/40">
                            <td className="p-2.5 font-sans font-medium text-slate-300">{item.filename}</td>
                            <td className="p-2.5 text-slate-400">{item.actual_label}</td>
                            <td className="p-2.5">
                              <span className={item.predicted_verdict === 'AUTHENTIC' ? 'text-emerald-400' : 'text-rose-400'}>
                                {item.predicted_verdict}
                              </span>
                            </td>
                            <td className="p-2.5 text-white font-bold">{item.authenticity_score}%</td>
                            <td className="p-2.5">
                              <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px]">
                                {item.status}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}
