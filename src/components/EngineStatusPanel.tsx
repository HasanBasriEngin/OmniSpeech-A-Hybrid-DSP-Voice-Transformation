/**
 * EngineStatusPanel
 *
 * Dönüşüm sonrasında hangi motorun (FreeVC / RVC / OpenCV) aktif
 * çalıştığını, hangisinin fallback'e düştüğünü ve kullanılan modelin
 * lisans/izin bilgisini gösterir.
 *
 * İlker Tugberk Evren — UI motor göstergesi görevi
 */

import type { EngineStatus, ModelMetadata } from "@/types/omni";

interface EngineStatusPanelProps {
  engineStatus: EngineStatus;
  modelMetadata: ModelMetadata;
  visible: boolean;
}

function StatusBadge({ active, label }: { active: boolean; label: string }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "6px",
        padding: "4px 10px",
        borderRadius: "6px",
        background: active ? "rgba(34,211,176,0.12)" : "rgba(255,255,255,0.05)",
        border: `1px solid ${active ? "rgba(34,211,176,0.4)" : "rgba(255,255,255,0.1)"}`,
        fontSize: "11px",
        fontWeight: 600,
        letterSpacing: "0.03em",
        color: active ? "#22d3b0" : "rgba(255,255,255,0.35)",
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: active ? "#22d3b0" : "rgba(255,255,255,0.2)",
          boxShadow: active ? "0 0 6px #22d3b0" : "none",
          flexShrink: 0,
        }}
      />
      {label}
      <span style={{ marginLeft: 2, opacity: 0.6 }}>{active ? "aktif" : "pasif"}</span>
    </div>
  );
}

export function EngineStatusPanel({ engineStatus, modelMetadata, visible }: EngineStatusPanelProps) {
  if (!visible) {
    return null;
  }

  const freevcActive = engineStatus.freevc_engine >= 0.5;
  const rvcActive = engineStatus.rvc_engine >= 0.5;
  const opencvActive = engineStatus.opencv_spectrogram_applied >= 0.5;
  const fallbackUsed = engineStatus.fallback_used;

  return (
    <div
      style={{
        marginTop: 12,
        padding: "12px 14px",
        borderRadius: 10,
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "rgba(255,255,255,0.4)",
          marginBottom: 10,
        }}
      >
        Motor Durumu
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        <StatusBadge active={opencvActive} label="OpenCV" />
        <StatusBadge active={freevcActive} label="FreeVC" />
        <StatusBadge active={rvcActive} label="RVC" />

        {fallbackUsed && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "4px 10px",
              borderRadius: 6,
              background: "rgba(251,191,36,0.1)",
              border: "1px solid rgba(251,191,36,0.3)",
              fontSize: 11,
              fontWeight: 600,
              color: "#fbbf24",
            }}
          >
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#fbbf24" }} />
            Fallback kullanıldı
          </div>
        )}
      </div>

      {/* Model metadata satırı */}
      {modelMetadata.is_licensed_profile && (
        <div
          style={{
            marginTop: 10,
            paddingTop: 10,
            borderTop: "1px solid rgba(255,255,255,0.06)",
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(255,255,255,0.4)" }}>
            Model Bilgisi
          </div>
          {modelMetadata.model_id && (
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "rgba(255,255,255,0.6)" }}>
              <span>Model ID</span>
              <strong style={{ color: "#e2e8f0" }}>{modelMetadata.model_id}</strong>
            </div>
          )}
          {modelMetadata.license && (
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "rgba(255,255,255,0.6)" }}>
              <span>Lisans</span>
              <strong style={{ color: "#e2e8f0" }}>{modelMetadata.license}</strong>
            </div>
          )}
          {modelMetadata.consent_owner && (
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "rgba(255,255,255,0.6)" }}>
              <span>İzin sahibi</span>
              <strong style={{ color: "#22d3b0" }}>{modelMetadata.consent_owner}</strong>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
