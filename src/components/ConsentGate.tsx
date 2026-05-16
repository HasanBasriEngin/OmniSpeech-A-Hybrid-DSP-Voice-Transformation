/**
 * ConsentGate
 *
 * RVC veya izinli profil modülü seçildiğinde gösterilen onay bileşeni.
 * Kullanıcı onay vermeden dönüşüm başlamaz.
 *
 * Harici API ile ses çekimi yasaktır. Bu bileşen bunu kullanıcıya
 * açıkça bildirir.
 *
 * İlker Tugberk Evren — consent katmanı görevi
 */

interface ConsentGateProps {
  visible: boolean;
  confirmed: boolean;
  onConfirm: (value: boolean) => void;
  /** "speaker" | "celebrity" — hangi modül için gösteriliyor */
  context?: "speaker" | "celebrity";
}

export function ConsentGate({ visible, confirmed, onConfirm, context = "celebrity" }: ConsentGateProps) {
  if (!visible) {
    return null;
  }

  const id = "consent-checkbox-gate";

  return (
    <div
      style={{
        margin: "12px 0",
        padding: "12px 14px",
        borderRadius: 10,
        background: confirmed ? "rgba(34,211,176,0.05)" : "rgba(248,113,113,0.06)",
        border: `1px solid ${confirmed ? "rgba(34,211,176,0.25)" : "rgba(248,113,113,0.25)"}`,
        transition: "background 0.2s, border-color 0.2s",
      }}
    >
      {/* Başlık */}
      <div
        style={{
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: confirmed ? "#22d3b0" : "#f87171",
          marginBottom: 8,
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        <span>🔒</span>
        İzin Gerekli
      </div>

      {/* Açıklama metni */}
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.6)", lineHeight: 1.6, marginBottom: 10 }}>
        {context === "speaker"
          ? "Konuşmacı klonu özelliği yalnızca izinli, yerel referans dosyalarıyla çalışır."
          : "İzinli profil modülü yalnızca yerel ve rıza alınmış ses modelleriyle çalışır."}
        {" "}
        <span style={{ color: "#f87171", fontWeight: 600 }}>
          Harici API ile ses çekimi, TTS veya cloud sesler kesinlikle yasaktır.
        </span>
      </div>

      {/* Onay satırı */}
      <label
        htmlFor={id}
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: 9,
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <input
          checked={confirmed}
          id={id}
          onChange={(e) => onConfirm(e.target.checked)}
          style={{
            marginTop: 2,
            accentColor: "#22d3b0",
            width: 14,
            height: 14,
            flexShrink: 0,
            cursor: "pointer",
          }}
          type="checkbox"
        />
        <span style={{ fontSize: 12, color: "rgba(255,255,255,0.8)", lineHeight: 1.5 }}>
          Bu referans/model için gerekli izinlere sahibim. Kullandığım ses dosyaları
          yerel, rıza alınmış kaynaklara aittir.
        </span>
      </label>

      {/* Onay verilmemişse uyarı */}
      {!confirmed && (
        <div
          style={{
            marginTop: 8,
            fontSize: 11,
            color: "#f87171",
            fontWeight: 500,
          }}
        >
          ⚠ Onay verilmeden dönüşüm başlamaz.
        </div>
      )}
    </div>
  );
}
