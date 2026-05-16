export type ConversionTask = "emotion" | "gender_age" | "speaker_clone" | "singing" | "celebrity";

export interface ConversionResult {
  outputPath: string;
  metrics: Record<string, number>;
  engine_status?: EngineStatus;
  model_metadata?: ModelMetadata;
}

/**
 * Motor durum bilgisi — backend'den her dönüşüm sonucunda gelir.
 * UI bu değerlere göre aktif/pasif gösterge render eder.
 */
export interface EngineStatus {
  /** 1.0 = FreeVC aktif kullanıldı, 0.0 = fallback/pasif */
  freevc_engine: number;
  /** 1.0 = RVC aktif kullanıldı, 0.0 = fallback/pasif */
  rvc_engine: number;
  /** 1.0 = OpenCV spektrogram ön-işleme uygulandı, 0.0 = atlandı */
  opencv_spectrogram_applied: number;
  /** Herhangi bir motor fallback'e düştüyse true */
  fallback_used: boolean;
}

/**
 * Çıktıda kullanılan modelin lisans ve izin bilgisi.
 */
export interface ModelMetadata {
  model_id: string | null;
  license: string | null;
  consent_owner: string | null;
  is_licensed_profile: boolean;
}

/** Boş/varsayılan motor durumu — henüz dönüşüm yapılmadığında kullanılır. */
export const DEFAULT_ENGINE_STATUS: EngineStatus = {
  freevc_engine: 0,
  rvc_engine: 0,
  opencv_spectrogram_applied: 0,
  fallback_used: false,
};
