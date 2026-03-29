import { invoke } from "@tauri-apps/api/core";

import type { ConversionResult, ConversionTask } from "@/types/omni";

const tauriOnly = async <T>(command: string, args?: Record<string, unknown>): Promise<T> => {
  return invoke<T>(command, args);
};

export const api = {
  startBackend: () => tauriOnly<string>("start_backend"),
  stopBackend: () => tauriOnly<string>("stop_backend"),
  backendHealth: () => tauriOnly<boolean>("backend_health"),
  pickAudioFile: () => tauriOnly<string | null>("pick_audio_file"),
  pickReferenceFiles: () => tauriOnly<string[]>("pick_reference_files"),
  pickMidiFile: () => tauriOnly<string | null>("pick_midi_file"),
  convertGenderAge: (inputPath: string, mode: string, outputPath: string | null = null) =>
    tauriOnly<ConversionResult>("convert_gender_age", { inputPath, mode, outputPath }),
  convertSpeakerClone: (inputPath: string, referencePaths: string[], outputPath: string | null = null) =>
    tauriOnly<ConversionResult>("convert_speaker_clone", { inputPath, referencePaths, outputPath }),
  convertSinging: (
    inputPath: string,
    midiPath: string | null,
    pitchContour: number[] | null = null,
    outputPath: string | null = null,
  ) => tauriOnly<ConversionResult>("convert_singing", { inputPath, midiPath, pitchContour, outputPath }),
  listVirtualMics: () => tauriOnly<string[]>("list_virtual_mic_devices"),
  startLiveSession: (
    task: ConversionTask,
    options: Record<string, unknown>,
    routeToVirtualMic: boolean,
    virtualMicDevice: string | null,
  ) => tauriOnly<string>("start_live_session", { task, options, routeToVirtualMic, virtualMicDevice }),
  processLiveChunk: (sessionId: string, chunk: number[]) =>
    tauriOnly<number[]>("process_live_chunk", { sessionId, chunk }),
  stopLiveSession: (sessionId: string) => tauriOnly<string>("stop_live_session", { sessionId }),
};
