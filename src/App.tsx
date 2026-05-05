import { convertFileSrc } from "@tauri-apps/api/core";
import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

import { api } from "@/lib/tauri";

type ModuleKey = "emotion" | "gender" | "speaker" | "singing";
type EmotionKey = "sad" | "angry" | "excited" | "whisper" | "calm";
type GenderMode = "male_to_female" | "female_to_male" | "adult_to_child" | "adult_to_elderly" | "child_to_adult";
type NavigationKey = "workspace" | "evaluation" | "settings";
type InputMode = "file" | "mic";
type FeatureTab = "f0" | "mfcc" | "energy";

type LogEntry = {
  time: string;
  text: string;
  pending?: boolean;
};

type ConversionPayload = {
  outputPath?: string;
  output_path?: string;
  metrics?: Record<string, number>;
};

type WaveformData = {
  peaks: number[];
  duration: number;
};

const TOTAL_DURATION = 2.8;
const LIVE_SAMPLE_RATE = 22050;

const moduleMeta: Record<ModuleKey, { label: string }> = {
  emotion: { label: "Emotion" },
  gender: { label: "Gender / Age" },
  speaker: { label: "Speaker / Clone" },
  singing: { label: "Singing Voice" },
};

const emotionDescriptions: Record<EmotionKey, string> = {
  sad: "Lower pitch and softer phrasing",
  angry: "Sharper attack and stronger energy",
  excited: "Higher pitch and faster delivery",
  whisper: "Breathier tone with reduced energy",
  calm: "Balanced tone with smoother pacing",
};

const genderModeLabels: Record<GenderMode, string> = {
  male_to_female: "Male to Female",
  female_to_male: "Female to Male",
  adult_to_child: "Adult to Child",
  adult_to_elderly: "Adult to Elderly",
  child_to_adult: "Child to Adult",
};

const genderModeDescriptions: Record<GenderMode, string> = {
  male_to_female: "Higher pitch with a brighter vocal tract",
  female_to_male: "Lower pitch with a deeper vocal tract",
  adult_to_child: "Higher pitch and smaller vocal shape",
  adult_to_elderly: "Softer pitch shift with aged timbre",
  child_to_adult: "Lower pitch and fuller adult tone",
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

function getAudioContextCtor() {
  return window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
}

function createAudioContext(sampleRate?: number) {
  const AudioContextCtor = getAudioContextCtor();
  if (!AudioContextCtor) {
    throw new Error("AudioContext is not available in this webview");
  }

  if (sampleRate) {
    try {
      return new AudioContextCtor({ sampleRate });
    } catch {
      return new AudioContextCtor();
    }
  }

  return new AudioContextCtor();
}

function resampleLinear(input: Float32Array, fromRate: number, toRate: number) {
  if (fromRate === toRate || input.length === 0) {
    return new Float32Array(input);
  }

  const ratio = toRate / fromRate;
  const outputLength = Math.max(1, Math.round(input.length * ratio));
  const output = new Float32Array(outputLength);

  for (let index = 0; index < outputLength; index += 1) {
    const sourceIndex = index / ratio;
    const left = Math.floor(sourceIndex);
    const right = Math.min(left + 1, input.length - 1);
    const mix = sourceIndex - left;
    output[index] = (input[left] ?? 0) * (1 - mix) + (input[right] ?? 0) * mix;
  }

  return output;
}

async function buildWaveform(path: string, pointCount = 520): Promise<WaveformData> {
  const response = await fetch(convertFileSrc(path));
  if (!response.ok) {
    throw new Error(`Could not read audio file (${response.status})`);
  }

  const context = createAudioContext();
  try {
    const bytes = await response.arrayBuffer();
    const decoded = await context.decodeAudioData(bytes.slice(0));
    const channel = decoded.getChannelData(0);
    const bucketSize = Math.max(1, Math.floor(channel.length / pointCount));
    const peaks: number[] = [];

    for (let bucket = 0; bucket < pointCount; bucket += 1) {
      const start = bucket * bucketSize;
      const end = Math.min(channel.length, start + bucketSize);
      let peak = 0;
      for (let index = start; index < end; index += 1) {
        peak = Math.max(peak, Math.abs(channel[index] ?? 0));
      }
      peaks.push(peak);
    }

    const maxPeak = Math.max(...peaks, 1e-5);
    return {
      peaks: peaks.map((peak) => peak / maxPeak),
      duration: decoded.duration,
    };
  } finally {
    if (context.state !== "closed") {
      await context.close();
    }
  }
}

function drawWave(canvas: HTMLCanvasElement | null, waveform: WaveformData | null, colorA: string, colorB: string, playhead: number | null) {
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

  if (!waveform || waveform.peaks.length === 0) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(canvas.width, mid);
    ctx.stroke();
    return;
  }

  const lineGradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
  lineGradient.addColorStop(0, colorA);
  lineGradient.addColorStop(1, colorB);
  const fillGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  fillGradient.addColorStop(0, `${colorA}42`);
  fillGradient.addColorStop(0.5, `${colorB}16`);
  fillGradient.addColorStop(1, `${colorA}42`);

  const peaks = waveform.peaks;
  const step = canvas.width / Math.max(1, peaks.length - 1);
  ctx.beginPath();
  peaks.forEach((peak, index) => {
    const x = index * step;
    const y = mid - peak * 36;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  for (let index = peaks.length - 1; index >= 0; index -= 1) {
    const x = index * step;
    const y = mid + peaks[index] * 36;
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = fillGradient;
  ctx.fill();

  ctx.beginPath();
  peaks.forEach((peak, index) => {
    const x = index * step;
    const y = mid - peak * 32;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.strokeStyle = lineGradient;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  if (playhead !== null) {
    const x = Math.max(0, Math.min(1, playhead)) * canvas.width;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
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
  const minutes = Math.floor(safe / 60);
  const whole = Math.floor(safe % 60);
  const frac = Math.floor((safe - Math.floor(safe)) * 10);
  return `${minutes}:${whole.toString().padStart(2, "0")}.${frac}`;
}

function manualPitchHz(pitchSteps: number) {
  return 220 * 2 ** (pitchSteps / 12);
}

function getOutputPath(result: ConversionPayload) {
  return result.outputPath ?? result.output_path ?? null;
}

function mergeAudioChunks(chunks: Float32Array[]) {
  const totalLength = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function writeAscii(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function encodeWav(samples: Float32Array, sampleRate: number) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  samples.forEach((sample) => {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
    offset += bytesPerSample;
  });

  return new Uint8Array(buffer);
}

export default function App() {
  const [activeNav, setActiveNav] = useState<NavigationKey>("workspace");
  const [activeModule, setActiveModule] = useState<ModuleKey>("emotion");
  const [selectedEmotion, setSelectedEmotion] = useState<EmotionKey>("angry");
  const [selectedGenderMode, setSelectedGenderMode] = useState<GenderMode>("male_to_female");
  const [inputMode, setInputMode] = useState<InputMode>("file");
  const [featureTab, setFeatureTab] = useState<FeatureTab>("f0");

  const [sourceFile, setSourceFile] = useState<string | null>(null);
  const [referenceFiles, setReferenceFiles] = useState<string[]>([]);
  const [midiFile, setMidiFile] = useState<string | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [originalWaveform, setOriginalWaveform] = useState<WaveformData | null>(null);
  const [processedWaveform, setProcessedWaveform] = useState<WaveformData | null>(null);
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
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordedFile, setRecordedFile] = useState<string | null>(null);
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
  const outputAudioRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const liveSessionRef = useRef<string | null>(null);
  const liveStreamRef = useRef<MediaStream | null>(null);
  const liveAudioContextRef = useRef<AudioContext | null>(null);
  const liveProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const liveSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const liveProcessingRef = useRef(false);
  const recordingStreamRef = useRef<MediaStream | null>(null);
  const recordingAudioContextRef = useRef<AudioContext | null>(null);
  const recordingProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const recordingSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const recordingChunksRef = useRef<Float32Array[]>([]);
  const recordingSampleRateRef = useRef(44100);
  const recordingStartedAtRef = useRef(0);
  const recordingTimerRef = useRef<number | null>(null);

  const addLog = (text: string, pending = false) => {
    setLogEntries((prev) => [...prev, { time: nowClock(), text, pending }].slice(-100));
  };

  const moduleButtons = useMemo(() => (Object.keys(moduleMeta) as ModuleKey[]).map((key) => ({ key, ...moduleMeta[key] })), []);
  const activeTask = useMemo(
    () =>
      activeModule === "emotion"
        ? "emotion"
        : activeModule === "speaker"
          ? "speaker_clone"
          : activeModule === "singing"
            ? "singing"
            : "gender_age",
    [activeModule],
  );

  const metricValues = useMemo(() => {
    const processingSeconds = metrics.processing_seconds ?? 1.4;
    const latencyMs = Math.max(1, Math.round(processingSeconds * 1000));
    const fidelity = metrics.snr_estimate_db ? Math.max(1, Math.min(5, metrics.snr_estimate_db / 8.5 + 2.2)) : 4.2;
    const intelligibility = metrics.output_median_f0 ? Math.max(1, Math.min(5, 3.2 + Math.abs((metrics.output_median_f0 - (metrics.input_median_f0 ?? 0)) / 400))) : 3.8;
    return { latencyMs, processingSeconds, fidelity, intelligibility };
  }, [metrics]);

  const redraw = () => {
    const originalPlayhead = sourceFile && !outputPath ? progress : null;
    const processedPlayhead = outputPath ? progress : null;
    drawWave(waveformOrigRef.current, originalWaveform, "#6c63ff", "#22d3b0", originalPlayhead);
    drawWave(waveformProcRef.current, processedWaveform, "#22d3b0", "#a78bfa", processedPlayhead);
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

  const stopBackend = async () => {
    await stopLive(false);
    try {
      await api.stopBackend();
      setBackendReady(false);
      addLog("Backend stopped");
    } catch (err) {
      addLog(`Backend stop failed: ${normalizeError(err)}`, true);
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
        stopPlayback();
        setOutputPath(null);
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

  const stopRecording = async (save = true) => {
    const chunks = [...recordingChunksRef.current];

    recordingProcessorRef.current?.disconnect();
    recordingSourceRef.current?.disconnect();
    recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (recordingAudioContextRef.current && recordingAudioContextRef.current.state !== "closed") {
      await recordingAudioContextRef.current.close();
    }
    if (recordingTimerRef.current !== null) {
      window.clearInterval(recordingTimerRef.current);
    }

    recordingProcessorRef.current = null;
    recordingSourceRef.current = null;
    recordingStreamRef.current = null;
    recordingAudioContextRef.current = null;
    recordingTimerRef.current = null;
    recordingChunksRef.current = [];
    setIsRecording(false);

    if (!save) {
      setRecordingSeconds(0);
      return null;
    }

    const totalSamples = chunks.reduce((total, chunk) => total + chunk.length, 0);
    if (totalSamples === 0) {
      addLog("Recording did not capture audio", true);
      return null;
    }

    try {
      const wavBytes = encodeWav(mergeAudioChunks(chunks), recordingSampleRateRef.current);
      const path = await api.saveRecordingWav(Array.from(wavBytes));
      stopPlayback();
      setOutputPath(null);
      setSourceFile(path);
      setRecordedFile(path);
      setInputMode("mic");
      setRecordingSeconds(Number(((Date.now() - recordingStartedAtRef.current) / 1000).toFixed(1)));
      addLog(`Mic recording saved: ${basename(path)}`);
      return path;
    } catch (err) {
      addLog(`Recording save failed: ${normalizeError(err)}`, true);
      return null;
    }
  };

  const startRecording = async () => {
    if (isRecording) {
      return;
    }

    if (isLive) {
      await stopLive(false);
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      addLog("Microphone recording is not available in this webview", true);
      return;
    }

    try {
      await stopRecording(false);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const context = createAudioContext();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);

      recordingChunksRef.current = [];
      recordingSampleRateRef.current = context.sampleRate;
      recordingStartedAtRef.current = Date.now();
      setRecordedFile(null);
      setRecordingSeconds(0);
      setInputMode("mic");

      processor.onaudioprocess = (event) => {
        event.outputBuffer.getChannelData(0).fill(0);
        recordingChunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };

      source.connect(processor);
      processor.connect(context.destination);

      recordingStreamRef.current = stream;
      recordingAudioContextRef.current = context;
      recordingSourceRef.current = source;
      recordingProcessorRef.current = processor;
      recordingTimerRef.current = window.setInterval(() => {
        setRecordingSeconds((Date.now() - recordingStartedAtRef.current) / 1000);
      }, 150);
      setIsRecording(true);
      addLog("Mic recording started");
    } catch (err) {
      await stopRecording(false);
      addLog(`Mic recording failed: ${normalizeError(err)}`, true);
    }
  };

  const stopLiveCapture = async () => {
    liveProcessorRef.current?.disconnect();
    liveSourceRef.current?.disconnect();
    liveStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (liveAudioContextRef.current && liveAudioContextRef.current.state !== "closed") {
      await liveAudioContextRef.current.close();
    }
    liveProcessorRef.current = null;
    liveSourceRef.current = null;
    liveStreamRef.current = null;
    liveAudioContextRef.current = null;
    liveProcessingRef.current = false;
  };

  const startLiveCapture = async (sessionId: string) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      addLog("Microphone capture is not available in this webview", true);
      return;
    }

    await stopLiveCapture();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const context = createAudioContext(LIVE_SAMPLE_RATE);
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(2048, 1, 1);

    if (context.sampleRate !== LIVE_SAMPLE_RATE) {
      addLog(`Live input resampled ${Math.round(context.sampleRate)}Hz -> ${LIVE_SAMPLE_RATE}Hz`);
    }

    processor.onaudioprocess = (event) => {
      event.outputBuffer.getChannelData(0).fill(0);
      if (liveProcessingRef.current || liveSessionRef.current !== sessionId) {
        return;
      }

      liveProcessingRef.current = true;
      const liveChunk = resampleLinear(event.inputBuffer.getChannelData(0), context.sampleRate, LIVE_SAMPLE_RATE);
      const chunk = Array.from(liveChunk);
      void api
        .processLiveChunk(sessionId, chunk)
        .catch((err) => addLog(`Live audio chunk failed: ${normalizeError(err)}`, true))
        .finally(() => {
          liveProcessingRef.current = false;
        });
    };

    source.connect(processor);
    processor.connect(context.destination);

    liveStreamRef.current = stream;
    liveAudioContextRef.current = context;
    liveSourceRef.current = source;
    liveProcessorRef.current = processor;
    addLog("Microphone stream connected");
  };

  const stopLive = async (log = true) => {
    const sessionToStop = liveSessionRef.current ?? liveSessionId;
    liveSessionRef.current = null;
    await stopLiveCapture();

    if (sessionToStop) {
      try {
        await api.stopLiveSession(sessionToStop);
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

    if (isRecording) {
      await stopRecording(false);
    }

    const ready = await ensureBackend();
    if (!ready) {
      return;
    }

    let startedSessionId: string | null = null;
    try {
      const sessionId = await api.startLiveSession(
        activeTask,
        {
          emotion: selectedEmotion,
          mode: selectedGenderMode,
          midi_path: midiFile,
          pitch_contour: activeModule === "singing" && !midiFile ? [manualPitchHz(pitchValue)] : null,
          reference_paths: referenceFiles,
          pitch_override: pitchValue,
          rate_override: rateValue,
          energy_override: energyValue,
        },
        routeToVirtualMic,
        selectedVirtualMic,
      );

      startedSessionId = sessionId;
      liveSessionRef.current = sessionId;
      setLiveSessionId(sessionId);
      setIsLive(true);
      setInputMode("mic");
      addLog(`Live session started (${activeTask})`);
      await startLiveCapture(sessionId);
    } catch (err) {
      if (startedSessionId) {
        try {
          await api.stopLiveSession(startedSessionId);
        } catch {
          // The start failure is the useful message for the user-facing log.
        }
      }
      liveSessionRef.current = null;
      setLiveSessionId(null);
      setIsLive(false);
      await stopLiveCapture();
      addLog(`Live start failed: ${normalizeError(err)}`, true);
    }
  };

  const stopPlayback = () => {
    const audio = outputAudioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    setPlaying(false);
    setProgress(0);
  };

  const loadPlaybackAudio = (path: string) => {
    const audio = outputAudioRef.current;
    if (!audio) {
      return;
    }

    audio.pause();
    audio.src = convertFileSrc(path);
    audio.currentTime = 0;
    audio.load();
    setPlaying(false);
    setProgress(0);
  };

  const togglePlayback = async () => {
    const playbackPath = outputPath ?? sourceFile;
    if (!playbackPath) {
      addLog("Once bir ses dosyasi sec veya MIC ile kayit al.", true);
      return;
    }

    const audio = outputAudioRef.current;
    if (!audio) {
      addLog("Audio player is not ready", true);
      return;
    }

    const src = convertFileSrc(playbackPath);
    if (audio.src !== src) {
      audio.src = src;
      audio.load();
    }

    if (!audio.paused) {
      audio.pause();
      return;
    }

    if (audio.ended || (Number.isFinite(audio.duration) && audio.currentTime >= audio.duration)) {
      audio.currentTime = 0;
    }

    try {
      await audio.play();
      addLog(`${outputPath ? "Playing processed audio" : "Playing original audio"}: ${basename(playbackPath)}`);
    } catch (err) {
      addLog(`Playback failed: ${normalizeError(err)}`, true);
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
      let result: ConversionPayload;

      if (activeTask === "emotion") {
        result = await api.convertEmotion(inputPath, selectedEmotion, pitchValue, rateValue, energyValue, null);
      } else if (activeTask === "gender_age") {
        result = await api.convertGenderAge(inputPath, selectedGenderMode, null);
      } else if (activeTask === "speaker_clone") {
        let refs = referenceFiles;
        if (refs.length === 0) {
          refs = await pickReferences();
        }
        if (refs.length === 0) {
          throw new Error("Speaker clone requires reference files");
        }
        result = await api.convertSpeakerClone(inputPath, refs, null);
      } else {
        const pitchContour = midiFile ? null : [manualPitchHz(pitchValue)];
        result = await api.convertSinging(inputPath, midiFile, pitchContour, null);
      }

      const nextOutputPath = getOutputPath(result);
      if (!nextOutputPath) {
        throw new Error("Backend did not return an output audio path");
      }

      setOutputPath(nextOutputPath);
      loadPlaybackAudio(nextOutputPath);
      setMetrics(result.metrics ?? {});
      if (typeof result.metrics?.output_duration_seconds === "number" && result.metrics.output_duration_seconds > 0) {
        setPlaybackDuration(result.metrics.output_duration_seconds);
      }

      addLog(`Output generated: ${basename(nextOutputPath)}`);
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
    const audio = outputAudioRef.current;
    if (audio && Number.isFinite(audio.duration) && audio.duration > 0) {
      audio.currentTime = next * audio.duration;
    }
  };

  useEffect(() => {
    let cancelled = false;
    if (!sourceFile) {
      setOriginalWaveform(null);
      return () => {
        cancelled = true;
      };
    }

    void buildWaveform(sourceFile)
      .then((waveform) => {
        if (cancelled) {
          return;
        }
        setOriginalWaveform(waveform);
        if (!outputPath) {
          setPlaybackDuration(waveform.duration);
        }
      })
      .catch((err) => addLog(`Original waveform failed: ${normalizeError(err)}`, true));

    return () => {
      cancelled = true;
    };
  }, [sourceFile, outputPath]);

  useEffect(() => {
    let cancelled = false;
    if (!outputPath) {
      setProcessedWaveform(null);
      return () => {
        cancelled = true;
      };
    }

    void buildWaveform(outputPath)
      .then((waveform) => {
        if (cancelled) {
          return;
        }
        setProcessedWaveform(waveform);
        setPlaybackDuration(waveform.duration);
      })
      .catch((err) => addLog(`Processed waveform failed: ${normalizeError(err)}`, true));

    return () => {
      cancelled = true;
    };
  }, [outputPath]);

  useEffect(() => {
    redraw();
  }, [pitchValue, rateValue, energyValue, featureTab, progress, activeModule, originalWaveform, processedWaveform, sourceFile, outputPath]);

  useEffect(() => {
    const onResize = () => redraw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [pitchValue, rateValue, energyValue, featureTab, progress, activeModule, originalWaveform, processedWaveform, sourceFile, outputPath]);

  useEffect(() => {
    void ensureBackend();
    void refreshVirtualMics();

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      void stopRecording(false);
      void stopLive(false);
    };
  }, []);

  const statusText = isLive ? "live" : backendReady ? "ready" : "offline";
  const statusColor = isLive || backendReady ? "#22d3b0" : "#f87171";
  const modeInfo =
    activeModule === "emotion"
      ? emotionDescriptions[selectedEmotion]
      : activeModule === "gender"
        ? genderModeDescriptions[selectedGenderMode]
        : activeModule === "singing"
          ? midiFile
            ? `MIDI: ${basename(midiFile)}`
            : `Manual target: ${Math.round(manualPitchHz(pitchValue))} Hz`
          : referenceFiles.length
            ? `${referenceFiles.length} reference file${referenceFiles.length > 1 ? "s" : ""}`
            : "Reference files required";
  const speakerReferencesRequired = activeModule === "speaker";
  const singingMidiSuggested = activeModule === "singing";
  const convertBlockedReason = speakerReferencesRequired && referenceFiles.length === 0 ? "Speaker / Clone icin en az bir referans dosyasi sec." : null;
  const convertButtonLabel = isConverting ? "Processing..." : convertBlockedReason ? "References Required" : "Convert Audio";

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
          <div className="tag">{moduleMeta[activeModule].label}</div>
          <div className="tag">{inputMode === "mic" ? "Mic Input" : "File Input"}</div>
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
              if (isLive) {
                void stopLive(false);
              }
              setActiveNav("workspace");
              setActiveModule(item.key);
              addLog(`Module selected: ${item.label}`);
            }}
            type="button"
          >
            <span className="nav-icon">M</span>
            {item.label}
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
                  <button
                    className={`btn-xs ${inputMode === "file" ? "active" : ""}`}
                    onClick={() => {
                      setInputMode("file");
                      if (isLive) {
                        void stopLive();
                      }
                      if (isRecording) {
                        void stopRecording(false);
                      }
                    }}
                    type="button"
                  >
                    FILE
                  </button>
                  <button className={`btn-xs ${inputMode === "mic" ? "active" : ""}`} onClick={() => setInputMode("mic")} type="button">
                    MIC
                  </button>
                </div>
              </div>

              <div className={`drop-zone ${sourceFile ? "loaded" : ""} ${inputMode === "mic" ? "static" : ""}`} onClick={() => void (inputMode === "file" ? pickSource() : undefined)}>
                <div className="drop-zone-title">
                  {inputMode === "file"
                    ? sourceFile
                      ? `${basename(sourceFile)} loaded`
                      : "Drop audio file here"
                    : isLive
                      ? "Microphone live session running"
                      : recordedFile
                        ? `${basename(recordedFile)} ready`
                        : "Microphone recorder"}
                </div>
                <div className="drop-zone-sub">
                  {inputMode === "file"
                    ? sourceFile
                      ? "Click to pick another file"
                      : "or click to browse"
                    : isLive
                      ? `Session: ${liveSessionId?.slice(0, 8) ?? "active"}`
                      : recordedFile
                        ? "Use Convert Audio or record again"
                        : "Press Record to capture a WAV input"}
                </div>
                <div className="fmt-tags">
                  <span className="fmt-tag">WAV</span>
                  <span className="fmt-tag">MP3</span>
                  <span className="fmt-tag">FLAC</span>
                  <span className="fmt-tag">16kHz/22kHz</span>
                </div>
                {inputMode === "mic" ? (
                  <div className="recorder-panel" onClick={(event) => event.stopPropagation()}>
                    <div className={`recording-dot ${isRecording ? "active" : ""}`}></div>
                    <div className="recording-time">{recordingSeconds.toFixed(1)}s</div>
                    {isRecording ? (
                      <button className="btn-xs active" onClick={() => void stopRecording(true)} type="button">
                        Stop & Save
                      </button>
                    ) : (
                      <button className="btn-xs active" onClick={() => void startRecording()} type="button">
                        Record
                      </button>
                    )}
                    {recordedFile ? <span className="recording-file">{basename(recordedFile)}</span> : null}
                  </div>
                ) : null}
              </div>

              {(speakerReferencesRequired || singingMidiSuggested) ? (
                <div className="action-row contextual">
                  {speakerReferencesRequired ? (
                    <button className="btn-xs active" onClick={() => void pickReferences()} type="button">
                      {referenceFiles.length ? "Change References" : "Select References"}
                    </button>
                  ) : null}
                  {singingMidiSuggested ? (
                    <button className="btn-xs active" onClick={() => void pickMidi()} type="button">
                      {midiFile ? "Change MIDI" : "Select MIDI"}
                    </button>
                  ) : null}
                </div>
              ) : null}

              <div className={`mini-note ${convertBlockedReason ? "warn" : ""}`}>
                {convertBlockedReason
                  ? convertBlockedReason
                  : activeModule === "emotion"
                    ? `${selectedEmotion} emotion profili secili ve donusume hazir.`
                    : activeModule === "gender"
                      ? `${genderModeLabels[selectedGenderMode]} modu secili ve donusume hazir.`
                      : singingMidiSuggested
                        ? midiFile
                          ? "Singing modu secilen MIDI melodisini kullanacak."
                          : "Singing modu pitch slider'indan hedef nota uretecek."
                        : `${referenceFiles.length} referans dosyasi hazir.`}
              </div>

              <div className="selection-strip">
                <div className="selection-chip active">
                  <span className="selection-label">Module</span>
                  <span className="selection-value">{moduleMeta[activeModule].label}</span>
                </div>
                <div className={`selection-chip ${sourceFile ? "ok" : ""}`}>
                  <span className="selection-label">Source</span>
                  <span className="selection-value">{sourceFile ? basename(sourceFile) : "Not selected"}</span>
                </div>
                <div className={`selection-chip ${referenceFiles.length > 0 || activeModule === "gender" || activeModule === "emotion" ? "ok" : speakerReferencesRequired ? "warn" : ""}`}>
                  <span className="selection-label">
                    {activeModule === "emotion" ? "Emotion" : activeModule === "gender" ? "Mode" : "References"}
                  </span>
                  <span className="selection-value">
                    {activeModule === "emotion"
                      ? selectedEmotion
                      : activeModule === "gender"
                        ? genderModeLabels[selectedGenderMode]
                        : referenceFiles.length > 0
                          ? `${referenceFiles.length} file${referenceFiles.length > 1 ? "s" : ""}`
                          : "None"}
                  </span>
                </div>
                <div className={`selection-chip ${midiFile ? "ok" : singingMidiSuggested ? "active" : ""}`}>
                  <span className="selection-label">MIDI</span>
                  <span className="selection-value">{midiFile ? basename(midiFile) : singingMidiSuggested ? "Suggested" : "Optional"}</span>
                </div>
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
                <button className="play-btn" disabled={!outputPath && !sourceFile} onClick={() => void togglePlayback()} type="button">
                  {playing ? "||" : ">"}
                </button>
                <div className="progress-wrap">
                  <div className="progress-bg" onClick={seekTo}>
                    <div className="progress-fill" style={{ width: `${progress * 100}%` }}></div>
                    <div className="progress-thumb" style={{ left: `${progress * 100}%` }}></div>
                  </div>
                </div>
                <span className="time-text">{formatTime(progress * playbackDuration)} / {formatTime(playbackDuration)}</span>
                <audio
                  onEnded={() => {
                    setPlaying(false);
                    setProgress(1);
                  }}
                  onLoadedMetadata={(event) => {
                    const duration = event.currentTarget.duration;
                    if (Number.isFinite(duration) && duration > 0) {
                      setPlaybackDuration(duration);
                    }
                  }}
                  onPause={() => setPlaying(false)}
                  onPlay={() => setPlaying(true)}
                  onTimeUpdate={(event) => {
                    const audio = event.currentTarget;
                    if (Number.isFinite(audio.duration) && audio.duration > 0) {
                      setPlaybackDuration(audio.duration);
                      setProgress(Math.max(0, Math.min(1, audio.currentTime / audio.duration)));
                    }
                  }}
                  preload="metadata"
                  ref={outputAudioRef}
                />
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
        ) : activeNav === "evaluation" ? (
          <>
            <section className="card">
              <div className="section-header">
                <span className="section-title">// evaluation</span>
              </div>
              <div className="status-grid">
                <div className="status-tile">
                  <span className="selection-label">Task</span>
                  <span className="selection-value">{moduleMeta[activeModule].label}</span>
                </div>
                <div className="status-tile">
                  <span className="selection-label">Mode</span>
                  <span className="selection-value">{modeInfo}</span>
                </div>
                <div className="status-tile">
                  <span className="selection-label">Input</span>
                  <span className="selection-value">{sourceFile ? basename(sourceFile) : inputMode === "mic" ? "Microphone" : "Not selected"}</span>
                </div>
                <div className="status-tile">
                  <span className="selection-label">Output</span>
                  <span className="selection-value">{outputPath ? basename(outputPath) : "No conversion yet"}</span>
                </div>
              </div>
            </section>

            <section className="card">
              <div className="section-header">
                <span className="section-title">// metrics_detail</span>
              </div>
              <div className="detail-list">
                {Object.entries(metrics).length ? (
                  Object.entries(metrics).map(([key, value]) => (
                    <div className="detail-row" key={key}>
                      <span>{key}</span>
                      <strong>{Number.isFinite(value) ? value.toFixed(3) : value}</strong>
                    </div>
                  ))
                ) : (
                  <div className="empty-state">Run a conversion to populate backend metrics.</div>
                )}
              </div>
            </section>
          </>
        ) : (
          <>
            <section className="card">
              <div className="section-header">
                <span className="section-title">// settings</span>
              </div>
              <div className="settings-grid">
                <div className="detail-row">
                  <span>Backend</span>
                  <strong>{backendReady ? "ready" : "offline"}</strong>
                </div>
                <div className="detail-row">
                  <span>Live session</span>
                  <strong>{isLive ? liveSessionId?.slice(0, 8) ?? "active" : "stopped"}</strong>
                </div>
                <div className="detail-row">
                  <span>Virtual mic devices</span>
                  <strong>{virtualMicDevices.length}</strong>
                </div>
                <div className="settings-actions">
                  <button className="btn-xs active" onClick={() => void ensureBackend()} type="button">Start Backend</button>
                  <button className="btn-xs" onClick={() => void stopBackend()} type="button">Stop Backend</button>
                </div>
              </div>
            </section>
          </>
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
          <div className="panel-title">// PARAMETERS</div>

          {activeModule === "emotion" ? (
            <>
              <div className="param-row">
                <div className="param-header"><span className="param-name">Pitch Shift</span><span className="param-val">{pitchValue >= 0 ? "+" : ""}{pitchValue.toFixed(1)} st</span></div>
                <input type="range" min={-6} max={6} step={0.1} value={pitchValue} onChange={(e) => setPitchValue(Number(e.target.value))} />
              </div>

              <div className="param-row">
                <div className="param-header"><span className="param-name">Speech Rate</span><span className="param-val teal">{rateValue.toFixed(2)}x</span></div>
                <input type="range" className="teal" min={0.6} max={1.5} step={0.01} value={rateValue} onChange={(e) => setRateValue(Number(e.target.value))} />
              </div>

              <div className="param-row">
                <div className="param-header"><span className="param-name">Energy</span><span className="param-val amber">{energyValue.toFixed(2)}x</span></div>
                <input type="range" className="amber" min={0.2} max={2.0} step={0.05} value={energyValue} onChange={(e) => setEnergyValue(Number(e.target.value))} />
              </div>
            </>
          ) : null}

          {activeModule === "singing" ? (
            <div className="param-row">
              <div className="param-header">
                <span className="param-name">Manual Target</span>
                <span className="param-val">{Math.round(manualPitchHz(pitchValue))} Hz</span>
              </div>
              <input type="range" min={-12} max={12} step={0.1} value={pitchValue} onChange={(e) => setPitchValue(Number(e.target.value))} />
              <div className="mini-note">{midiFile ? "MIDI selected, manual target is ignored." : "Used when no MIDI file is selected."}</div>
            </div>
          ) : null}

          {activeModule === "speaker" ? (
            <div className={`mini-note ${referenceFiles.length ? "" : "warn"}`}>
              {referenceFiles.length ? `${referenceFiles.length} reference file${referenceFiles.length > 1 ? "s" : ""} selected.` : "Select reference files from the audio input area."}
            </div>
          ) : null}

          {activeModule === "gender" ? (
            <div>
              <div className="chip-row vertical">
                {(Object.keys(genderModeLabels) as GenderMode[]).map((mode) => (
                  <button
                    className={`chip wide ${selectedGenderMode === mode ? "selected" : ""}`}
                    key={mode}
                    onClick={() => {
                      setSelectedGenderMode(mode);
                      addLog(`Gender mode set: ${genderModeLabels[mode]}`);
                    }}
                    type="button"
                  >
                    {genderModeLabels[mode]}
                  </button>
                ))}
              </div>
              <div className="mini-note">{genderModeDescriptions[selectedGenderMode]}</div>
            </div>
          ) : null}

          {activeModule === "emotion" ? (
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
            <div className="mini-note">{modeInfo}</div>
          </div>
          ) : null}
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

        <button className="convert-btn" disabled={Boolean(convertBlockedReason) || isConverting} onClick={() => void runConvert()} type="button">
          {convertButtonLabel}
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
