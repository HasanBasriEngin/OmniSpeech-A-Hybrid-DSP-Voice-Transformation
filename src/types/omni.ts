export type ConversionTask = "emotion" | "gender_age" | "speaker_clone" | "singing" | "celebrity" | "voice_clone";

export interface ConversionResult {
  outputPath: string;
  metrics: Record<string, number>;
}
