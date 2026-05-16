import { convertFileSrc } from "@tauri-apps/api/core";
import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

import { api } from "@/lib/tauri";
import { EngineStatusPanel } from "@/components/EngineStatusPanel";
import { ConsentGate } from "@/components/ConsentGate";
import type { EngineStatus, ModelMetadata } from "@/types/omni";
import { DEFAULT_ENGINE_STATUS } from "@/types/omni";

type ModuleKey = "emotion" | "gender" | "speaker" | "singing" | "celebrity";
type EmotionKey = "sad" | "angry" | "excited" | "whisper" | "calm";
type CelebrityKey = "michael_jackson" | "morgan_freeman" | "adele" | "james_earl_jones" | "taylor_swift";
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
  engine_status?: EngineStatus;
  model_metadata?: ModelMetadata;
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

const moduleMeta: Record<ModuleKey, { label: string; labelTr: string }> = {
  emotion:   { label: "Emotion",          labelTr: "Duygu Dönüşümü" },
  gender:    { label: "Gender / Age",     labelTr: "Cinsiyet / Yaş" },
  speaker:   { label: "Speaker / Clone",  labelTr: "Konuşmacı Klonu" },
  singing:   { label: "Singing Voice",    labelTr: "Şarkı Sesi" },
  celebrity: { label: "Celebrity Voice",  labelTr: "Ünlü Sesi" },
};

const emotionDescriptions: Record<EmotionKey, string> = {
  sad:     "Daha düşük perde ve yumuşak ifade",
  angry:   "Sert atak ve güçlü enerji",
  excited: "Yüksek perde ve hızlı konuşma",
  whisper: "Nefesli ton ve düşük enerji",
  calm:    "Dengeli ton ve akıcı ritim",
};

const celebrityLabels: Record<CelebrityKey, string> = {
  michael_jackson: "Michael Jackson",
  morgan_freeman: "Morgan Freeman",
  adele: "Adele",
  james_earl_jones: "James Earl Jones",
  taylor_swift: "Taylor Swift",
};

const genderModeLabels: Record<GenderMode, string> = {
  male_to_female:   "Erkek → Kadın",
  female_to_male:   "Kadın → Erkek",
  adult_to_child:   "Yetişkin → Çocuk",
  adult_to_elderly: "Yetişkin → Yaşlı",
  child_to_adult:   "Çocuk → Yetişkin",
};

const genderModeDescriptions: Record<GenderMode, string> = {
  male_to_female:   "Daha yüksek perde ve parlak vokal yolu",
  female_to_male:   "Daha düşük perde ve derin vokal yolu",
  adult_to_child:   "Yüksek perde ve küçük vokal şekli",
  adult_to_elderly: "Yaşlı timbr ile yumuşak perde kayması",
  child_to_adult:   "Düşük perde ve dolgun yetişkin tonu",
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

function delay(milliseconds: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

async function waitForBackendReady(timeoutMs = 10_000) {
  const startedAt = Date.now();
  const delays = [100, 150, 200, 300, 400, 500, 750, 1_000];
  let attempt = 0;

  while (Date.now() - startedAt < timeoutMs) {
    try {
      if (await api.backendHealth()) {
        return true;
      }
    } catch {
      // Backend may still be binding the port; keep polling until timeout.
    }

    await delay(delays[Math.min(attempt, delays.length - 1)]);
    attempt += 1;
  }

  try {
    return await api.backendHealth();
  } catch {
    return false;
  }
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
  const [selectedCelebrity, setSelectedCelebrity] = useState<CelebrityKey>("michael_jackson");
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
  const [engineStatus, setEngineStatus] = useState<EngineStatus>(DEFAULT_ENGINE_STATUS);
  const [modelMetadata, setModelMetadata] = useState<ModelMetadata>({ model_id: null, license: null, consent_owner: null, is_licensed_profile: false });
  const [consentConfirmed, setConsentConfirmed] = useState(false);

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
    { time: "14:32", text: "Arayüz hazır" },
    { time: "14:33", text: "Başlamak için kaynak ses seç", pending: true },
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
            : activeModule === "celebrity"
              ? "celebrity"
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
      addLog("Backend hazirlaniyor...", true);
      await api.startBackend();
      const healthy = await waitForBackendReady();
      setBackendReady(healthy);
      addLog(healthy ? "Backend hazır" : "Backend sağlık kontrolü başarısız", !healthy);
      return healthy;
    } catch (err) {
      addLog(`Backend hatası: ${normalizeError(err)}`, true);
      return false;
    }
  };

  const stopBackend = async () => {
    await stopLive(false);
    try {
      await api.stopBackend();
      setBackendReady(false);
      addLog("Backend durduruldu");
    } catch (err) {
      addLog(`Backend durdurulamadı: ${normalizeError(err)}`, true);
    }
  };

  const refreshVirtualMics = async () => {
    try {
      const devices = await api.listVirtualMics();
      setVirtualMicDevices(devices);
      if (selectedVirtualMic && !devices.includes(selectedVirtualMic)) {
        setSelectedVirtualMic(null);
      }
      addLog(`Sanal mikrofonlar yenilendi (${devices.length})`);
    } catch (err) {
      addLog(`Sanal mikrofon yenilenemedi: ${normalizeError(err)}`, true);
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
        addLog(`Kaynak seçildi: ${basename(selected)}`);
      }
      return selected;
    } catch (err) {
      addLog(`Kaynak seçimi başarısız: ${normalizeError(err)}`, true);
      return null;
    }
  };

  const pickReferences = async () => {
    try {
      const selected = await api.pickReferenceFiles();
      setReferenceFiles(selected);
      addLog(selected.length ? `Referans seçildi (${selected.length})` : "Referanslar temizlendi");
      return selected;
    } catch (err) {
      addLog(`Referans seçimi başarısız: ${normalizeError(err)}`, true);
      return [];
    }
  };

  const pickMidi = async () => {
    try {
      const selected = await api.pickMidiFile();
      setMidiFile(selected);
      addLog(selected ? `MIDI seçildi: ${basename(selected)}` : "MIDI temizlendi");
      return selected;
    } catch (err) {
      addLog(`MIDI seçimi başarısız: ${normalizeError(err)}`, true);
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
      addLog("Kayıt ses yakalayamadı", true);
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
      addLog(`Mikrofon kaydı kaydedildi: ${basename(path)}`);
      return path;
    } catch (err) {
      addLog(`Kayıt kaydedilemedi: ${normalizeError(err)}`, true);
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
      addLog("Bu pencerede mikrofon kaydı kullanılamıyor", true);
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
      addLog("Mikrofon kaydı başladı");
    } catch (err) {
      await stopRecording(false);
      addLog(`Mikrofon kaydı başarısız: ${normalizeError(err)}`, true);
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
      addLog("Bu pencerede mikrofon yakalama kullanılamıyor", true);
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
      addLog(`Canlı giriş yeniden örneklendi ${Math.round(context.sampleRate)}Hz -> ${LIVE_SAMPLE_RATE}Hz`);
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
        .catch((err) => addLog(`Canlı ses parçası işlenemedi: ${normalizeError(err)}`, true))
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
    addLog("Mikrofon akışı bağlandı");
  };

  const stopLive = async (log = true) => {
    const sessionToStop = liveSessionRef.current ?? liveSessionId;
    liveSessionRef.current = null;
    await stopLiveCapture();

    if (sessionToStop) {
      try {
        await api.stopLiveSession(sessionToStop);
      } catch (err) {
        addLog(`Canlı oturum durdurulamadı: ${normalizeError(err)}`, true);
      }
    }

    setLiveSessionId(null);
    setIsLive(false);
    if (log) {
      addLog("Canlı oturum durduruldu");
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
          celebrity: selectedCelebrity,
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
      addLog(`Canlı oturum başladı (${activeTask})`);
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
      addLog(`Canlı oturum başlatılamadı: ${normalizeError(err)}`, true);
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
      addLog("Önce bir ses dosyası seç veya mikrofonla kayıt al.", true);
      return;
    }

    const audio = outputAudioRef.current;
    if (!audio) {
      addLog("Ses çalar hazır değil", true);
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
      addLog(`${outputPath ? "İşlenmiş ses çalınıyor" : "Orijinal ses çalınıyor"}: ${basename(playbackPath)}`);
    } catch (err) {
      addLog(`Ses çalma başarısız: ${normalizeError(err)}`, true);
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
      addLog("Önce kaynak ses dosyası seç", true);
      return;
    }

    const ready = await ensureBackend();
    if (!ready) {
      return;
    }

    setIsConverting(true);
    addLog(`Dönüşüm başladı: ${moduleMeta[activeModule].labelTr}`, true);

    try {
      let result: ConversionPayload;

      if (activeTask === "emotion") {
        result = await api.convertEmotion(inputPath, selectedEmotion, pitchValue, rateValue, energyValue, null);
      } else if (activeTask === "gender_age") {
        result = await api.convertGenderAge(inputPath, selectedGenderMode, null);
      } else if (activeTask === "celebrity") {
        result = await api.convertCelebrity(inputPath, selectedCelebrity, null);
      } else if (activeTask === "speaker_clone") {
        let refs = referenceFiles;
        if (refs.length === 0) {
          refs = await pickReferences();
        }
        if (refs.length === 0) {
          throw new Error("Konuşmacı klonu için referans dosya gerekli");
        }
        result = await api.convertSpeakerClone(inputPath, refs, null);
      } else {
        const pitchContour = midiFile ? null : [manualPitchHz(pitchValue)];
        result = await api.convertSinging(inputPath, midiFile, pitchContour, null);
      }

      const nextOutputPath = getOutputPath(result);
      if (!nextOutputPath) {
        throw new Error("Backend çıktı ses yolu döndürmedi");
      }

      setOutputPath(nextOutputPath);
      loadPlaybackAudio(nextOutputPath);
      setMetrics(result.metrics ?? {});
      if (result.engine_status) setEngineStatus(result.engine_status);
      if (result.model_metadata) setModelMetadata(result.model_metadata);
      if (typeof result.metrics?.output_duration_seconds === "number" && result.metrics.output_duration_seconds > 0) {
        setPlaybackDuration(result.metrics.output_duration_seconds);
      }

      addLog(`Çıktı oluşturuldu: ${basename(nextOutputPath)}`);
    } catch (err) {
      addLog(`Dönüşüm başarısız: ${normalizeError(err)}`, true);
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
      .catch((err) => addLog(`Orijinal dalga formu çıkarılamadı: ${normalizeError(err)}`, true));

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
      .catch((err) => addLog(`İşlenmiş dalga formu çıkarılamadı: ${normalizeError(err)}`, true));

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

  const statusText = isLive ? "canlı" : backendReady ? "hazır" : "çevrimdışı";
  const statusColor = isLive || backendReady ? "#22d3b0" : "#f87171";
  const modeInfo =
    activeModule === "emotion"
      ? emotionDescriptions[selectedEmotion]
      : activeModule === "gender"
        ? genderModeDescriptions[selectedGenderMode]
        : activeModule === "singing"
          ? midiFile
            ? `MIDI: ${basename(midiFile)}`
            : `Manuel hedef: ${Math.round(manualPitchHz(pitchValue))} Hz`
          : activeModule === "celebrity"
            ? `Profil: ${celebrityLabels[selectedCelebrity]}`
            : referenceFiles.length
              ? `${referenceFiles.length} referans dosyası`
              : "Referans dosyası gerekli";
  const speakerReferencesRequired = activeModule === "speaker";
  const singingMidiSuggested = activeModule === "singing";
  const consentRequired = activeModule === "celebrity" || activeModule === "speaker";
  const convertBlockedReason =
    (consentRequired && !consentConfirmed)
      ? "Bu modül için izin onayı gereklidir."
      : speakerReferencesRequired && referenceFiles.length === 0
        ? "Konuşmacı Klonu için en az bir referans dosyası seç."
        : null;
  const convertButtonLabel = isConverting ? "İşleniyor..." : convertBlockedReason ? "Referans Gerekli" : "Sesi Dönüştür";

  return (
    <div className="app">
      <header className="topbar">
        <div className="logo">
          <div className="logo-icon">S</div>
          <div>
            <div className="logo-text">SpeechWarp</div>
            <div className="logo-sub">v0.9 · CENG 384</div>
          </div>
        </div>
        <div className="topbar-right">
          <div className="status-dot" style={{ background: statusColor, boxShadow: `0 0 6px ${statusColor}` }}></div>
          <span className="status-text">{statusText}</span>
          <div className="tag">{moduleMeta[activeModule].labelTr}</div>
          <div className="tag">{inputMode === "mic" ? "Mikrofon" : "Dosya"}</div>
        </div>
      </header>

      <aside className="sidebar">
        <div className="sidebar-section">Gezinti</div>
        <button className={`nav-item nav-button ${activeNav === "workspace" ? "active" : ""}`} onClick={() => setActiveNav("workspace")} type="button">
          <span className="nav-icon">
            <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
          </span>
          Çalışma Alanı
        </button>

        <div className="sidebar-section">DSP Modülleri</div>
        {moduleButtons.map((item) => (
          <button
            className={`nav-item nav-button ${activeModule === item.key ? "active" : ""}`}
            key={item.key}
            onClick={() => {
              if (isLive) void stopLive(false);
              setActiveNav("workspace");
              setActiveModule(item.key);
              addLog(`Modül seçildi: ${item.labelTr}`);
            }}
            type="button"
          >
            <span className="nav-icon">
              {item.key === "emotion"   && <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8.5 14s1 2 3.5 2 3.5-2 3.5-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>}
              {item.key === "gender"    && <svg viewBox="0 0 24 24"><circle cx="10" cy="8" r="4"/><path d="M2 20c0-4 3.6-7 8-7"/><path d="M16 8h6m-3-3 3 3-3 3"/></svg>}
              {item.key === "speaker"   && <svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>}
              {item.key === "singing"   && <svg viewBox="0 0 24 24"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>}
              {item.key === "celebrity" && <svg viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>}
            </span>
            {item.labelTr}
          </button>
        ))}

        <div className="sidebar-section">Sistem</div>
        <button className={`nav-item nav-button ${activeNav === "evaluation" ? "active" : ""}`} onClick={() => setActiveNav("evaluation")} type="button">
          <span className="nav-icon">
            <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          </span>
          Değerlendirme
        </button>
        <button className={`nav-item nav-button ${activeNav === "settings" ? "active" : ""}`} onClick={() => setActiveNav("settings")} type="button">
          <span className="nav-icon">
            <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          </span>
          Ayarlar
        </button>
      </aside>

      <main className="main">
        {activeNav === "workspace" ? (
          <>
            <section className="card">
              <div className="section-header">
                <span className="section-title">Ses Girişi</span>
                <div className="section-controls">
                  <button
                    className={`btn-xs ${inputMode === "file" ? "active" : ""}`}
                    onClick={() => {
                      setInputMode("file");
                      if (isLive) void stopLive();
                      if (isRecording) void stopRecording(false);
                    }}
                    type="button"
                  >
                    DOSYA
                  </button>
                  <button className={`btn-xs ${inputMode === "mic" ? "active" : ""}`} onClick={() => setInputMode("mic")} type="button">
                    MİK
                  </button>
                </div>
              </div>

              <div className={`drop-zone ${sourceFile ? "loaded" : ""} ${inputMode === "mic" ? "static" : ""}`} onClick={() => void (inputMode === "file" ? pickSource() : undefined)}>
                <div className="drop-zone-title">
                  {inputMode === "file"
                    ? sourceFile
                      ? `${basename(sourceFile)} yüklendi`
                      : "Ses dosyasını buraya bırak"
                    : isLive
                      ? "Canlı mikrofon oturumu çalışıyor"
                      : recordedFile
                        ? `${basename(recordedFile)} hazır`
                        : "Mikrofon kaydedici"}
                </div>
                <div className="drop-zone-sub">
                  {inputMode === "file"
                    ? sourceFile
                      ? "Başka bir dosya seçmek için tıkla"
                      : "ya da dosya seçmek için tıkla"
                    : isLive
                      ? `Oturum: ${liveSessionId?.slice(0, 8) ?? "aktif"}`
                      : recordedFile
                        ? "Dönüştür veya yeniden kaydet"
                        : "WAV girdi yakalamak için Kaydet'e bas"}
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
                      <button className="btn-xs active" onClick={() => void stopRecording(true)} type="button">Dur & Kaydet</button>
                    ) : (
                      <button className="btn-xs active" onClick={() => void startRecording()} type="button">Kaydet</button>
                    )}
                    {recordedFile ? <span className="recording-file">{basename(recordedFile)}</span> : null}
                  </div>
                ) : null}
              </div>

              {(speakerReferencesRequired || singingMidiSuggested) ? (
                <div className="action-row contextual">
                  {speakerReferencesRequired ? (
                    <button className="btn-xs active" onClick={() => void pickReferences()} type="button">
                      {referenceFiles.length ? "Referansları Değiştir" : "Referans Seç"}
                    </button>
                  ) : null}
                  {singingMidiSuggested ? (
                    <button className="btn-xs active" onClick={() => void pickMidi()} type="button">
                      {midiFile ? "MIDI Değiştir" : "MIDI Seç"}
                    </button>
                  ) : null}
                </div>
              ) : null}

              <ConsentGate
                confirmed={consentConfirmed}
                context={activeModule === "speaker" ? "speaker" : "celebrity"}
                onConfirm={setConsentConfirmed}
                visible={consentRequired}
              />
              <div className={`mini-note ${convertBlockedReason ? "warn" : ""}`}>
                {convertBlockedReason
                  ? convertBlockedReason
                  : activeModule === "emotion"
                    ? `${selectedEmotion} duygu profili seçili ve dönüşüme hazır.`
                    : activeModule === "gender"
                      ? `${genderModeLabels[selectedGenderMode]} modu seçili ve dönüşüme hazır.`
                      : activeModule === "celebrity"
                        ? `${celebrityLabels[selectedCelebrity]} profili secili ve donusume hazir.`
                        : singingMidiSuggested
                          ? midiFile
                            ? "Şarkı modu seçilen MIDI melodisini kullanacak."
                            : "Şarkı modu pitch slider'ından hedef nota üretecek."
                          : `${referenceFiles.length} referans dosyası hazır.`}
              </div>

              <div className="selection-strip">
                <div className="selection-chip active">
                  <span className="selection-label">Modül</span>
                  <span className="selection-value">{moduleMeta[activeModule].labelTr}</span>
                </div>
                <div className={`selection-chip ${sourceFile ? "ok" : ""}`}>
                  <span className="selection-label">Kaynak</span>
                  <span className="selection-value">{sourceFile ? basename(sourceFile) : "Seçilmedi"}</span>
                </div>
                <div className={`selection-chip ${referenceFiles.length > 0 || activeModule === "gender" || activeModule === "emotion" || activeModule === "celebrity" ? "ok" : speakerReferencesRequired ? "warn" : ""}`}>
                  <span className="selection-label">
                    {activeModule === "emotion" ? "Duygu" : activeModule === "gender" ? "Mod" : activeModule === "celebrity" ? "Profil" : "Referanslar"}
                  </span>
                  <span className="selection-value">
                    {activeModule === "emotion"
                      ? selectedEmotion
                      : activeModule === "gender"
                        ? genderModeLabels[selectedGenderMode]
                        : activeModule === "celebrity"
                          ? celebrityLabels[selectedCelebrity]
                          : referenceFiles.length > 0
                            ? `${referenceFiles.length} dosya`
                            : "Yok"}
                  </span>
                </div>
                <div className={`selection-chip ${midiFile ? "ok" : singingMidiSuggested ? "active" : ""}`}>
                  <span className="selection-label">MIDI</span>
                  <span className="selection-value">{midiFile ? basename(midiFile) : singingMidiSuggested ? "Önerilen" : "İsteğe bağlı"}</span>
                </div>
              </div>

              <div className="waveform-area">
                <div className="waveform-label">orijinal</div>
                <div className="waveform-label right">{playbackDuration.toFixed(1)}s</div>
                <canvas className="waveform" ref={waveformOrigRef}></canvas>
              </div>

              <div className="waveform-area processed">
                <div className="waveform-label">işlenmiş</div>
                <canvas className="waveform" ref={waveformProcRef}></canvas>
              </div>

              <div className="playback-bar">
                <button className="play-btn" disabled={!outputPath && !sourceFile} onClick={() => void togglePlayback()} type="button">
                  {playing ? "❚❚" : "▶"}
                </button>
                <div className="progress-wrap">
                  <div className="progress-bg" onClick={seekTo}>
                    <div className="progress-fill" style={{ width: `${progress * 100}%` }}></div>
                    <div className="progress-thumb" style={{ left: `${progress * 100}%` }}></div>
                  </div>
                </div>
                <span className="time-text">{formatTime(progress * playbackDuration)} / {formatTime(playbackDuration)}</span>
                <audio
                  onEnded={() => { setPlaying(false); setProgress(1); }}
                  onLoadedMetadata={(event) => {
                    const duration = event.currentTarget.duration;
                    if (Number.isFinite(duration) && duration > 0) setPlaybackDuration(duration);
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
                <span className="section-title">Frekans Analizi</span>
                <div className="section-controls">
                  <button className={`btn-xs ${featureTab === "f0" ? "active" : ""}`} onClick={() => setFeatureTab("f0")} type="button">F0</button>
                  <button className={`btn-xs ${featureTab === "mfcc" ? "active" : ""}`} onClick={() => setFeatureTab("mfcc")} type="button">MFCC</button>
                  <button className={`btn-xs ${featureTab === "energy" ? "active" : ""}`} onClick={() => setFeatureTab("energy")} type="button">ENERJİ</button>
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
                <span className="section-title">Değerlendirme</span>
              </div>
              <div className="status-grid">
                <div className="status-tile">
                  <span className="selection-label">Görev</span>
                  <span className="selection-value">{moduleMeta[activeModule].labelTr}</span>
                </div>
                <div className="status-tile">
                  <span className="selection-label">Mod</span>
                  <span className="selection-value">{modeInfo}</span>
                </div>
                <div className="status-tile">
                  <span className="selection-label">Giriş</span>
                  <span className="selection-value">{sourceFile ? basename(sourceFile) : inputMode === "mic" ? "Mikrofon" : "Seçilmedi"}</span>
                </div>
                <div className="status-tile">
                  <span className="selection-label">Çıkış</span>
                  <span className="selection-value">{outputPath ? basename(outputPath) : "Henüz dönüşüm yok"}</span>
                </div>
              </div>
            </section>

            <section className="card">
              <div className="section-header">
                <span className="selection-title">Metrik Detayları</span>
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
                  <div className="empty-state">Backend metriklerini görmek için bir dönüşüm çalıştır.</div>
                )}
              </div>
              <EngineStatusPanel
                engineStatus={engineStatus}
                modelMetadata={modelMetadata}
                visible={true}
              />
            </section>
          </>
        ) : (
          <>
            <section className="card">
              <div className="section-header">
                <span className="section-title">Ayarlar</span>
              </div>
              <div className="settings-grid">
                <div className="detail-row">
                  <span>Backend</span>
                  <strong>{backendReady ? "hazır" : "çevrimdışı"}</strong>
                </div>
                <div className="detail-row">
                  <span>Canlı oturum</span>
                  <strong>{isLive ? liveSessionId?.slice(0, 8) ?? "aktif" : "durduruldu"}</strong>
                </div>
                <div className="detail-row">
                  <span>Sanal mikrofon aygıtları</span>
                  <strong>{virtualMicDevices.length}</strong>
                </div>
                <div className="settings-actions">
                  <button className="btn-xs active" onClick={() => void ensureBackend()} type="button">Backend Başlat</button>
                  <button className="btn-xs" onClick={() => void stopBackend()} type="button">Backend Durdur</button>
                </div>
              </div>
            </section>
          </>
        )}

        <section className="metric-grid">
          <div className="metric-card ok"><div className="metric-label">GECİKME</div><div className="metric-val">{metricValues.latencyMs} <span>ms</span></div></div>
          <div className="metric-card ok"><div className="metric-label">İŞLEM SÜRESİ</div><div className="metric-val">{metricValues.processingSeconds.toFixed(2)} <span>s</span></div></div>
          <div className="metric-card"><div className="metric-label">DOĞRULUK</div><div className="metric-val">{metricValues.fidelity.toFixed(1)} <span>/5</span></div></div>
          <div className="metric-card ok"><div className="metric-label">ANLAŞILIRLIK</div><div className="metric-val">{metricValues.intelligibility.toFixed(1)} <span>/5</span></div></div>
        </section>
      </main>

      <aside className="panel">
        <div>
          <div className="panel-title">Parametreler</div>

          {activeModule === "emotion" ? (
            <>
              <div className="param-row">
                <div className="param-header"><span className="param-name">Perde Kayması</span><span className="param-val">{pitchValue >= 0 ? "+" : ""}{pitchValue.toFixed(1)} st</span></div>
                <input type="range" min={-6} max={6} step={0.1} value={pitchValue} onChange={(e) => setPitchValue(Number(e.target.value))} />
              </div>
              <div className="param-row">
                <div className="param-header"><span className="param-name">Konuşma Hızı</span><span className="param-val teal">{rateValue.toFixed(2)}x</span></div>
                <input type="range" className="teal" min={0.6} max={1.5} step={0.01} value={rateValue} onChange={(e) => setRateValue(Number(e.target.value))} />
              </div>
              <div className="param-row">
                <div className="param-header"><span className="param-name">Enerji</span><span className="param-val amber">{energyValue.toFixed(2)}x</span></div>
                <input type="range" className="amber" min={0.2} max={2.0} step={0.05} value={energyValue} onChange={(e) => setEnergyValue(Number(e.target.value))} />
              </div>
            </>
          ) : null}

          {activeModule === "singing" ? (
            <div className="param-row">
              <div className="param-header">
                <span className="param-name">Manuel Hedef</span>
                <span className="param-val">{Math.round(manualPitchHz(pitchValue))} Hz</span>
              </div>
              <input type="range" min={-12} max={12} step={0.1} value={pitchValue} onChange={(e) => setPitchValue(Number(e.target.value))} />
              <div className="mini-note">{midiFile ? "MIDI seçili, manuel hedef göz ardı edilir." : "MIDI dosyası seçilmediğinde kullanılır."}</div>
            </div>
          ) : null}

          {activeModule === "speaker" ? (
            <div className={`mini-note ${referenceFiles.length ? "" : "warn"}`}>
              {referenceFiles.length ? `${referenceFiles.length} referans dosyası seçili.` : "Ses girişi alanından referans dosyaları seç."}
            </div>
          ) : null}

          {activeModule === "celebrity" ? (
            <div>
              <div className="mini-note">
                Ünlü ses klonu modülü aktif. Kaynak ses dosyasını seçip dönüştür.
              </div>
              <div className="panel-title" style={{ marginTop: "10px" }}>Hedef Profil</div>
              <div className="chip-row vertical">
                {(Object.keys(celebrityLabels) as CelebrityKey[]).map((celebrity) => (
                  <button
                    className={`chip wide ${selectedCelebrity === celebrity ? "selected" : ""}`}
                    key={celebrity}
                    onClick={() => {
                      setSelectedCelebrity(celebrity);
                      addLog(`Ünlü profili: ${celebrityLabels[celebrity]}`);
                    }}
                    type="button"
                  >
                    {celebrityLabels[celebrity]}
                  </button>
                ))}
              </div>
              <div className="mini-note">{modeInfo}</div>
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
                      addLog(`Cinsiyet modu: ${genderModeLabels[mode]}`);
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
              <div className="panel-title" style={{ marginTop: "10px" }}>Hedef Duygu</div>
              <div className="chip-row">
                {(["sad", "angry", "excited", "whisper", "calm"] as EmotionKey[]).map((emotion) => (
                  <button
                    className={`chip ${selectedEmotion === emotion ? `sel-${emotion}` : ""}`}
                    key={emotion}
                    onClick={() => {
                      setSelectedEmotion(emotion);
                      addLog(`Duygu hedefi: ${emotion}`);
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
          <div className="panel-title">Canlı Mod</div>
          <div className="check-row">
            <input checked={routeToVirtualMic} id="route-vmic" onChange={(e) => setRouteToVirtualMic(e.target.checked)} type="checkbox" />
            <label htmlFor="route-vmic">Sanal mikrofona yönlendir</label>
          </div>
          <div className="panel-inline">
            <select className="select-field" value={selectedVirtualMic ?? ""} onChange={(e) => setSelectedVirtualMic(e.target.value || null)}>
              <option value="">Sistem varsayılan çıkışı</option>
              {virtualMicDevices.map((device) => (
                <option key={device} value={device}>{device}</option>
              ))}
            </select>
            <button className="btn-xs" onClick={() => void refreshVirtualMics()} type="button">Yenile</button>
          </div>
          <button className="btn-xs" onClick={() => void startLive()} type="button">
            {isLive ? "Canlı Oturumu Durdur" : "Canlı Oturum Başlat"}
          </button>
        </div>

        <button className="convert-btn" disabled={Boolean(convertBlockedReason) || isConverting} onClick={() => void runConvert()} type="button">
          {convertButtonLabel}
        </button>

        <div>
          <div className="panel-title">Oturum Günlüğü</div>
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
