export type ConversionTask = "emotion" | "gender_age" | "speaker_clone" | "singing" | "celebrity";

export interface ConversionResult {
  outputPath: string;
  metrics: Record<string, number>;
}
