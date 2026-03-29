import { create } from "zustand";

import type { ConversionTask } from "@/types/omni";

interface OmniStore {
  backendReady: boolean;
  task: ConversionTask;
  genderMode: string;
  sourceFile: string | null;
  referenceFiles: string[];
  midiFile: string | null;
  outputPath: string | null;
  metrics: Record<string, number>;
  isBusy: boolean;
  virtualMicDevices: string[];
  selectedVirtualMic: string | null;
  routeToVirtualMic: boolean;
  liveSessionId: string | null;
  logs: string[];
  setBackendReady: (ready: boolean) => void;
  setTask: (task: ConversionTask) => void;
  setGenderMode: (mode: string) => void;
  setSourceFile: (path: string | null) => void;
  setReferenceFiles: (paths: string[]) => void;
  setMidiFile: (path: string | null) => void;
  setOutput: (path: string | null, metrics?: Record<string, number>) => void;
  setBusy: (busy: boolean) => void;
  setVirtualMicDevices: (devices: string[]) => void;
  setSelectedVirtualMic: (device: string | null) => void;
  setRouteToVirtualMic: (route: boolean) => void;
  setLiveSessionId: (sessionId: string | null) => void;
  addLog: (entry: string) => void;
}

export const useOmniStore = create<OmniStore>((set, get) => ({
  backendReady: false,
  task: "gender_age",
  genderMode: "male_to_female",
  sourceFile: null,
  referenceFiles: [],
  midiFile: null,
  outputPath: null,
  metrics: {},
  isBusy: false,
  virtualMicDevices: [],
  selectedVirtualMic: null,
  routeToVirtualMic: false,
  liveSessionId: null,
  logs: [],
  setBackendReady: (backendReady) => set({ backendReady }),
  setTask: (task) => set({ task }),
  setGenderMode: (genderMode) => set({ genderMode }),
  setSourceFile: (sourceFile) => set({ sourceFile }),
  setReferenceFiles: (referenceFiles) => set({ referenceFiles }),
  setMidiFile: (midiFile) => set({ midiFile }),
  setOutput: (outputPath, metrics = {}) => set({ outputPath, metrics }),
  setBusy: (isBusy) => set({ isBusy }),
  setVirtualMicDevices: (virtualMicDevices) => set({ virtualMicDevices }),
  setSelectedVirtualMic: (selectedVirtualMic) => set({ selectedVirtualMic }),
  setRouteToVirtualMic: (routeToVirtualMic) => set({ routeToVirtualMic }),
  setLiveSessionId: (liveSessionId) => set({ liveSessionId }),
  addLog: (entry) => {
    const next = [...get().logs, `[${new Date().toLocaleTimeString()}] ${entry}`].slice(-60);
    set({ logs: next });
  },
}));
