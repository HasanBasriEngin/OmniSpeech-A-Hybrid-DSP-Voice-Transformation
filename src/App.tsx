import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

import { api } from "@/lib/tauri";

type ModuleKey = "emotion" | "gender" | "speaker" | "singing";
type EmotionKey = "sad" | "angry" | "excited" | "whisper" | "calm";
type NavigationKey = "workspace" | "evaluation" | "settings";
type InputMode = "file" | "mic";
type FeatureTab = "f0" | "mfcc" | "energy";

type LogEntry = {
  time: string;
  text: string;
  pending?: boolean;
};

const TOTAL_DURATION = 2.8;

const moduleMeta: Record<ModuleKey, { label: string; badge: string }> = {
  emotion: { label: "Emotion", badge: "FR10-15" },
  gender: { label: "Gender / Age", badge: "FR16-21" },
  speaker: { label: "Speaker / Clone", badge: "FR22-28" },
  singing: { label: "Singing Voice", badge: "FR29-33" },
};

const emotionToMode: Record<EmotionKey, string> = {
  sad: "adult_to_elderly",
  angry: "female_to_male",
  excited: "adult_to_child",
  whisper: "male_to_female",
  calm: "child_to_adult",
};

function nowClock() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function basename(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

function normalizeError(err: unknown) {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err);
}

function drawWave(canvas: HTMLCanvasElement | null, seed: number, colorA: string, colorB: string, amp: number) {
  if (!canvas) {
    return;
  }

  const parent = canvas.parentElement;
  const width = Math.max(parent?.clientWidth ?? 600, 400);
  canvas.width = width;
  canvas.height = 88;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const mid = canvas.height / 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const lineGradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
  lineGradient.addColorStop(0, colorA);
  lineGradient.addColorStop(1, colorB);

  ctx.beginPath();
  for (let x = 0; x < canvas.width; x += 1) {
    const t = x / canvas.width;
    const y =
      mid +
      amp *
        (Math.sin(t * 28 + seed) * 0.5 +
          Math.sin(t * 71 + seed * 1.3) * 0.25 +
          Math.sin(t * 130 + seed * 0.7) * 0.12 +
          Math.sin(t * 9 + seed * 2) * 0.13) *
        Math.sin(t * Math.PI);

    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.strokeStyle = lineGradient;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawPitch(canvas: HTMLCanvasElement | null, tab: FeatureTab) {
  if (!canvas) {
    return;
  }

  const parent = canvas.parentElement;
  const width = Math.max(parent?.clientWidth ?? 600, 400);
  canvas.width = width;
  canvas.height = 70;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const a = tab === "mfcc" ? 6.2 : tab === "energy" ? 3.4 : 9.0;
  const b = tab === "mfcc" ? 10.3 : tab === "energy" ? 7.1 : 3.5;
  const c = tab === "mfcc" ? 17.8 : tab === "energy" ? 12.6 : 17.0;

  const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
  gradient.addColorStop(0, "#6c63ff");
  gradient.addColorStop(1, "#22d3b0");
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2;

  ctx.beginPath();
  for (let x = 0; x < canvas.width; x += 1) {
    const t = x / canvas.width;
    const y = 50 - 42 * (Math.sin(t * a) * 0.4 + Math.sin(t * b) * 0.35 + Math.sin(t * c) * 0.1) * Math.sin(t * Math.PI);
    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}

function formatTime(seconds: number) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const whole = Math.floor(safe);
  const frac = Math.floor((safe - whole) * 10);
  return `0:0${whole}.${frac}`;
}

export default function App() {
  const [activeNav, setActiveNav] = useState<NavigationKey>("workspace");
  const [activeModule, setActiveModule] = useState<ModuleKey>("emotion");
  const [selectedEmotion, setSelectedEmotion] = useState<EmotionKey>("angry");
  const [inputMode, setInputMode] = useState<InputMode>("file");
  const [featureTab, setFeatureTab] = useState<FeatureTab>("f0");

  const [sourceFile, setSourceFile] = useState<string | null>(null);
  const [referenceFiles, setReferenceFiles] = useState<string[]>([]);
  const [midiFile, setMidiFile] = useState<string | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Record<string, number>>({});

  const [pitchValue, setPitchValue] = useState(1.5);
  const [rateValue, setRateValue] = useState(1.15);
  const [energyValue, setEnergyValue] = useState(1.4);

  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0.34);
  const [playbackDuration, setPlaybackDuration] = useState(TOTAL_DURATION);

  const [backendReady, setBackendReady] = useState(false);
  const [isConverting, setIsConverting] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [liveSessionId, setLiveSessionId] = useState<string | null>(null);
  const [routeToVirtualMic, setRouteToVirtualMic] = useState(false);
  const [virtualMicDevices, setVirtualMicDevices] = useState<string[]>([]);
  const [selectedVirtualMic, setSelectedVirtualMic] = useState<string | null>(null);

  const [logEntries, setLogEntries] = useState<LogEntry[]>([
    { time: "14:32", text: "UI ready" },
    { time: "14:33", text: "Pick source audio to begin", pending: true },
  ]);

  const waveformOrigRef = useRef<HTMLCanvasElement | null>(null);
  const waveformProcRef = useRef<HTMLCanvasElement | null>(null);
  const pitchRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const addLog = (text: string, pending = false) => {
    setLogEntries((prev) => [...prev, { time: nowClock(), text, pending }].slice(-100));
  };

  const moduleButtons = useMemo(() => (Object.keys(moduleMeta) as ModuleKey[]).map((key) => ({ key, ...moduleMeta[key] })), []);

  const metricValues = useMemo(() => {
    const processingSeconds = metrics.processing_seconds ?? 1.4;
    const latencyMs = Math.max(1, Math.round(processingSeconds * 1000));
    const fidelity = metrics.snr_estimate_db ? Math.max(1, Math.min(5, metrics.snr_estimate_db / 8.5 + 2.2)) : 4.2;
    const intelligibility = metrics.output_median_f0 ? Math.max(1, Math.min(5, 3.2 + Math.abs((metrics.output_median_f0 - (metrics.input_median_f0 ?? 0)) / 400))) : 3.8;
    return { latencyMs, processingSeconds, fidelity, intelligibility };
  }, [metrics]);

  const redraw = () => {
    drawWave(waveformOrigRef.current, 1.0 + progress, "#6c63ff", "#22d3b0", 28);
    drawWave(waveformProcRef.current, 2.4 + pitchValue / 2 + rateValue + energyValue / 3, "#22d3b0", "#a78bfa", 22);
    drawPitch(pitchRef.current, featureTab);
  };

  const ensureBackend = async () => {
    if (backendReady) {
      return true;
    }

    try {
      await api.startBackend();
      const healthy = await api.backendHealth();
      setBackendReady(healthy);
      addLog(healthy ? "Backend ready" : "Backend health failed", !healthy);
      return healthy;
    } catch (err) {
      addLog(`Backend error: ${normalizeError(err)}`, true);
      return false;
    }
  };

  const refreshVirtualMics = async () => {
    try {
      const devices = await api.listVirtualMics();
      setVirtualMicDevices(devices);
      if (selectedVirtualMic && !devices.includes(selectedVirtualMic)) {
        setSelectedVirtualMic(null);
      }
      addLog(`Virtual mics refreshed (${devices.length})`);
    } catch (err) {
      addLog(`Virtual mic refresh failed: ${normalizeError(err)}`, true);
    }
  };

  const pickSource = async () => {
    try {
      const selected = await api.pickAudioFile();
      if (selected) {
        setSourceFile(selected);
        setInputMode("file");
        addLog(`Input selected: ${basename(selected)}`);
      }
      return selected;
    } catch (err) {
      addLog(`Input picker failed: ${normalizeError(err)}`, true);
      return null;
    }
  };

  const pickReferences = async () => {
    try {
      const selected = await api.pickReferenceFiles();
      setReferenceFiles(selected);
      addLog(selected.length ? `References selected (${selected.length})` : "References cleared");
      return selected;
    } catch (err) {
      addLog(`Reference picker failed: ${normalizeError(err)}`, true);
      return [];
    }
  };

  const pickMidi = async () => {
    try {
      const selected = await api.pickMidiFile();
      setMidiFile(selected);
      addLog(selected ? `MIDI selected: ${basename(selected)}` : "MIDI cleared");
      return selected;
    } catch (err) {
      addLog(`MIDI picker failed: ${normalizeError(err)}`, true);
      return null;
    }
  };

  const stopLive = async (log = true) => {
    if (liveSessionId) {
      try {
        await api.stopLiveSession(liveSessionId);
      } catch (err) {
        addLog(`Stop live failed: ${normalizeError(err)}`, true);
      }
    }

    setLiveSessionId(null);
    setIsLive(false);
    if (log) {
      addLog("Live session stopped");
    }
  };

  const startLive = async () => {
    if (isLive) {
      await stopLive();
      return;
    }

    const ready = await ensureBackend();
    if (!ready) {
      return;
    }

    try {
      const task = activeModule === "speaker" ? "speaker_clone" : activeModule === "singing" ? "singing" : "gender_age";
      const mode = activeModule === "emotion" ? emotionToMode[selectedEmotion] : "male_to_female";

      const sessionId = await api.startLiveSession(
        task,
        {
          mode,
          midi_path: midiFile,
          reference_paths: referenceFiles,
          pitch_ratio: pitchValue,
          speech_rate: rateValue,
          energy_envelope: energyValue,
        },
        routeToVirtualMic,
        selectedVirtualMic,
      );

      setLiveSessionId(sessionId);
      setIsLive(true);
      setInputMode("mic");
      addLog(`Live session started (${task})`);
    } catch (err) {
      addLog(`Live start failed: ${normalizeError(err)}`, true);
    }
  };

  const runConvert = async () => {
    if (isConverting) {
      return;
    }

    let inputPath = sourceFile;
    if (!inputPath) {
      inputPath = await pickSource();
    }
    if (!inputPath) {
      addLog("Please select input audio first", true);
      return;
    }

    const ready = await ensureBackend();
    if (!ready) {
      return;
    }

    setIsConverting(true);
    addLog(`Conversion started for ${activeModule}`, true);

    try {
      const task = activeModule === "speaker" ? "speaker_clone" : activeModule === "singing" ? "singing" : "gender_age";
      let result: { outputPath: string; metrics: Record<string, number> };

      if (task === "gender_age") {
        const mode = activeModule === "emotion" ? emotionToMode[selectedEmotion] : "male_to_female";
        result = await api.convertGenderAge(inputPath, mode, null);
      } else if (task === "speaker_clone") {
        let refs = referenceFiles;
        if (refs.length === 0) {
          refs = await pickReferences();
        }
        if (refs.length === 0) {
          throw new Error("Speaker clone requires reference files");
        }
        result = await api.convertSpeakerClone(inputPath, refs, null);
      } else {
        result = await api.convertSinging(inputPath, midiFile, null, null);
      }

      setOutputPath(result.outputPath);
      setMetrics(result.metrics ?? {});
      if (typeof result.metrics?.output_duration_seconds === "number" && result.metrics.output_duration_seconds > 0) {
        setPlaybackDuration(result.metrics.output_duration_seconds);
      }

      addLog(`Output generated: ${basename(result.outputPath)}`);
    } catch (err) {
      addLog(`Conversion failed: ${normalizeError(err)}`, true);
    } finally {
      setIsConverting(false);
    }
  };

  const seekTo = (event: MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const next = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    setProgress(next);
  };

  useEffect(() => {
    redraw();
  }, [pitchValue, rateValue, energyValue, featureTab, progress, activeModule]);

  useEffect(() => {
    const onResize = () => redraw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [pitchValue, rateValue, energyValue, featureTab, progress, activeModule]);

  useEffect(() => {
    if (!playing) {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      return;
    }

    const tick = () => {
      setProgress((prev) => {
        const next = prev + 1 / Math.max(30, playbackDuration * 60);
        if (next >= 1) {
          window.setTimeout(() => setPlaying(false), 0);
          return 1;
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [playing, playbackDuration]);

  useEffect(() => {
    void ensureBackend();
    void refreshVirtualMics();

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      void stopLive(false);
    };
  }, []);

  const statusText = isLive ? "live" : backendReady ? "ready" : "offline";
  const statusColor = isLive || backendReady ? "#22d3b0" : "#f87171";
  const modeInfo = activeModule === "emotion" ? emotionToMode[selectedEmotion] : "male_to_female";

  return (
    <div className="app">
      <header className="topbar">
        <div className="logo">
          <div className="logo-icon">S</div>
          <div>
            <div className="logo-text">SpeechWarp</div>
            <div className="logo-sub">v0.9 - CENG 384</div>
          </div>
        </div>
        <div className="topbar-right">
          <div className="status-dot" style={{ background: statusColor, boxShadow: `0 0 6px ${statusColor}` }}></div>
          <span className="status-text">{statusText}</span>
          <div className="tag">CPU - 12% load</div>
          <div className="tag">{isLive ? "Live Active" : "Desktop Mode"}</div>
        </div>
      </header>

      <aside className="sidebar">
        <div className="sidebar-section">Navigation</div>
        <button className={`nav-item nav-button ${activeNav === "workspace" ? "active" : ""}`} onClick={() => setActiveNav("workspace")} type="button">
          <span className="nav-icon">W</span>
          Workspace
        </button>

        <div className="sidebar-section">DSP Modules</div>
        {moduleButtons.map((item) => (
          <button
            className={`nav-item nav-button ${activeModule === item.key ? "active" : ""}`}
            key={item.key}
            onClick={() => {
              setActiveModule(item.key);
              addLog(`Module selected: ${item.label}`);
            }}
            type="button"
          >
            <span className="nav-icon">M</span>
            {item.label}
            <span className="nav-badge">{item.badge}</span>
          </button>
        ))}

        <div className="sidebar-section">System</div>
        <button className={`nav-item nav-button ${activeNav === "evaluation" ? "active" : ""}`} onClick={() => setActiveNav("evaluation")} type="button">
          <span className="nav-icon">E</span>
          Evaluation
        </button>
        <button className={`nav-item nav-button ${activeNav === "settings" ? "active" : ""}`} onClick={() => setActiveNav("settings")} type="button">
          <span className="nav-icon">S</span>
          Settings
        </button>
      </aside>

      <main className="main">
        {activeNav === "workspace" ? (
          <>
            <section className="card">
              <div className="section-header">
                <span className="section-title">// audio_input</span>
                <div className="section-controls">
                  <button className={`btn-xs ${inputMode === "file" ? "active" : ""}`} onClick={() => setInputMode("file")} type="button">
                    FILE
                  </button>
                  <button className={`btn-xs ${inputMode === "mic" ? "active" : ""}`} onClick={() => void startLive()} type="button">
                    MIC
                  </button>
                </div>
              </div>

              <div className={`drop-zone ${sourceFile ? "loaded" : ""}`} onClick={() => void (inputMode === "file" ? pickSource() : startLive())}>
                <div className="drop-zone-title">
                  {inputMode === "file"
                    ? sourceFile
                      ? `${basename(sourceFile)} loaded`
                      : "Drop audio file here"
                    : isLive
                      ? "Microphone live session running"
                      : "Start live microphone session"}
                </div>
                <div className="drop-zone-sub">
                  {inputMode === "file" ? (sourceFile ? "Click to pick another file" : "or click to browse") : isLive ? `Session: ${liveSessionId?.slice(0, 8) ?? "active"}` : "Click to start live mode"}
                </div>
                <div className="fmt-tags">
                  <span className="fmt-tag">WAV</span>
                  <span className="fmt-tag">MP3</span>
                  <span className="fmt-tag">FLAC</span>
                  <span className="fmt-tag">16kHz/22kHz</span>
                </div>
              </div>

              <div className="action-row">
                <button className="btn-xs" onClick={() => void pickSource()} type="button">Select Source</button>
                <button className="btn-xs" onClick={() => void pickReferences()} type="button">References</button>
                <button className="btn-xs" onClick={() => void pickMidi()} type="button">MIDI</button>
                <button className="btn-xs" onClick={() => void refreshVirtualMics()} type="button">Refresh VMic</button>
              </div>

              <div className="waveform-area">
                <div className="waveform-label">original</div>
                <div className="waveform-label right">{playbackDuration.toFixed(1)}s</div>
                <canvas className="waveform" ref={waveformOrigRef}></canvas>
              </div>

              <div className="waveform-area processed">
                <div className="waveform-label">processed</div>
                <canvas className="waveform" ref={waveformProcRef}></canvas>
              </div>

              <div className="playback-bar">
                <button className="play-btn" onClick={() => setPlaying((v) => !v)} type="button">
                  {playing ? "||" : ">"}
                </button>
                <div className="progress-wrap">
                  <div className="progress-bg" onClick={seekTo}>
                    <div className="progress-fill" style={{ width: `${progress * 100}%` }}></div>
                    <div className="progress-thumb" style={{ left: `${progress * 100}%` }}></div>
                  </div>
                </div>
                <span className="time-text">{formatTime(progress * playbackDuration)} / {formatTime(playbackDuration)}</span>
              </div>
            </section>

            <section className="card">
              <div className="section-header">
                <span className="section-title">// f0_pitch_contour</span>
                <div className="section-controls">
                  <button className={`btn-xs ${featureTab === "f0" ? "active" : ""}`} onClick={() => setFeatureTab("f0")} type="button">F0</button>
                  <button className={`btn-xs ${featureTab === "mfcc" ? "active" : ""}`} onClick={() => setFeatureTab("mfcc")} type="button">MFCC</button>
                  <button className={`btn-xs ${featureTab === "energy" ? "active" : ""}`} onClick={() => setFeatureTab("energy")} type="button">ENERGY</button>
                </div>
              </div>
              <div className="pitch-area">
                <canvas ref={pitchRef}></canvas>
              </div>
            </section>
          </>
        ) : (
          <section className="card">
            <div className="section-header">
              <span className="section-title">// {activeNav}</span>
            </div>
            <div className="action-row">
              <button className="btn-xs" onClick={() => void ensureBackend()} type="button">Start Backend</button>
              <button className="btn-xs" onClick={() => void api.stopBackend().then(() => { setBackendReady(false); addLog("Backend stopped"); })} type="button">Stop Backend</button>
              <button className="btn-xs" onClick={() => void refreshVirtualMics()} type="button">Refresh VMic</button>
              <button className="btn-xs" onClick={() => void startLive()} type="button">{isLive ? "Stop Live" : "Start Live"}</button>
            </div>
          </section>
        )}

        <section className="metric-grid">
          <div className="metric-card ok"><div className="metric-label">LATENCY</div><div className="metric-val">{metricValues.latencyMs} <span>ms</span></div></div>
          <div className="metric-card ok"><div className="metric-label">PROC TIME</div><div className="metric-val">{metricValues.processingSeconds.toFixed(2)} <span>s</span></div></div>
          <div className="metric-card"><div className="metric-label">FIDELITY</div><div className="metric-val">{metricValues.fidelity.toFixed(1)} <span>/5</span></div></div>
          <div className="metric-card ok"><div className="metric-label">INTELLIG.</div><div className="metric-val">{metricValues.intelligibility.toFixed(1)} <span>/5</span></div></div>
        </section>
      </main>

      <aside className="panel">
        <div>
          <div className="panel-title">// MODULE_SELECT</div>
          <div className="module-grid">
            {moduleButtons.map((item) => (
              <button
                className={`module-btn ${activeModule === item.key ? `active-${item.key}` : ""}`}
                key={`panel-${item.key}`}
                onClick={() => {
                  setActiveModule(item.key);
                  addLog(`Module selected: ${item.label}`);
                }}
                type="button"
              >
                <span className="module-btn-label">{item.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="panel-title">// PARAMETERS</div>

          <div className="param-row">
            <div className="param-header"><span className="param-name">Pitch Ratio</span><span className="param-val">{pitchValue >= 0 ? "+" : ""}{pitchValue.toFixed(1)} st</span></div>
            <input type="range" min={-6} max={6} step={0.1} value={pitchValue} onChange={(e) => setPitchValue(Number(e.target.value))} />
          </div>

          <div className="param-row">
            <div className="param-header"><span className="param-name">Speech Rate</span><span className="param-val teal">{rateValue.toFixed(2)}x</span></div>
            <input type="range" className="teal" min={0.6} max={1.5} step={0.01} value={rateValue} onChange={(e) => setRateValue(Number(e.target.value))} />
          </div>

          <div className="param-row">
            <div className="param-header"><span className="param-name">Energy Envelope</span><span className="param-val amber">{energyValue.toFixed(2)}x</span></div>
            <input type="range" className="amber" min={0.2} max={2.0} step={0.05} value={energyValue} onChange={(e) => setEnergyValue(Number(e.target.value))} />
          </div>

          <div>
            <div className="panel-title">// TARGET_EMOTION</div>
            <div className="chip-row">
              {(["sad", "angry", "excited", "whisper", "calm"] as EmotionKey[]).map((emotion) => (
                <button
                  className={`chip ${selectedEmotion === emotion ? `sel-${emotion}` : ""}`}
                  key={emotion}
                  onClick={() => {
                    setSelectedEmotion(emotion);
                    addLog(`Emotion target set: ${emotion}`);
                  }}
                  type="button"
                >
                  {emotion}
                </button>
              ))}
            </div>
            <div className="mini-note">Mapped preset: {modeInfo}</div>
          </div>
        </div>

        <div className="panel-live">
          <div className="panel-title">// LIVE</div>
          <div className="check-row">
            <input checked={routeToVirtualMic} id="route-vmic" onChange={(e) => setRouteToVirtualMic(e.target.checked)} type="checkbox" />
            <label htmlFor="route-vmic">Route to virtual mic</label>
          </div>

          <div className="panel-inline">
            <select className="select-field" value={selectedVirtualMic ?? ""} onChange={(e) => setSelectedVirtualMic(e.target.value || null)}>
              <option value="">System default output</option>
              {virtualMicDevices.map((device) => (
                <option key={device} value={device}>{device}</option>
              ))}
            </select>
            <button className="btn-xs" onClick={() => void refreshVirtualMics()} type="button">Refresh</button>
          </div>

          <button className="btn-xs" onClick={() => void startLive()} type="button">{isLive ? "Stop Live Session" : "Start Live Session"}</button>
        </div>

        <button className="convert-btn" onClick={() => void runConvert()} type="button">
          {isConverting ? "Processing..." : "Convert Audio"}
        </button>

        <div>
          <div className="panel-title">// SESSION_LOG</div>
          <div className="log-list">
            {logEntries.map((entry, index) => (
              <div className="log-entry" key={`${entry.time}-${index}`}>
                <div className={`log-dot ${entry.pending ? "pending" : ""}`}></div>
                <span className="log-time">{entry.time}</span>
                <span className="log-text">{entry.text}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
