import { GlassCard } from "@/components/GlassCard";
import type { ConversionTask } from "@/types/omni";

interface ConvertCardProps {
  task: ConversionTask;
  genderMode: string;
  isBusy: boolean;
  outputPath: string | null;
  metrics: Record<string, number>;
  onTaskChange: (task: ConversionTask) => void;
  onGenderModeChange: (mode: string) => void;
  onConvert: () => void;
}

const genderModes = [
  "male_to_female",
  "female_to_male",
  "adult_to_child",
  "adult_to_elderly",
  "child_to_adult",
];

export function ConvertCard({
  task,
  genderMode,
  isBusy,
  outputPath,
  metrics,
  onTaskChange,
  onGenderModeChange,
  onConvert,
}: ConvertCardProps) {
  return (
    <GlassCard title="Dönüşüm" subtitle="Dönüşüm modunu seç ve işlemi başlat.">
      <div className="grid gap-3 md:grid-cols-2">
        <label className="text-sm text-slate-300">
          Görev
          <select
            className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/70 px-3 py-2 text-sm"
            onChange={(event) => onTaskChange(event.target.value as ConversionTask)}
            value={task}
          >
            <option value="gender_age">Cinsiyet / Yaş</option>
            <option value="speaker_clone">Konuşmacı Klonu</option>
            <option value="singing">Konuşmadan Şarkıya</option>
          </select>
        </label>

        {task === "gender_age" ? (
          <label className="text-sm text-slate-300">
            Ön Ayar
            <select
              className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/70 px-3 py-2 text-sm"
              onChange={(event) => onGenderModeChange(event.target.value)}
              value={genderMode}
            >
              {genderModes.map((mode) => (
                <option key={mode} value={mode}>
                  {mode}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <div className="rounded-xl border border-slate-300/10 bg-slate-800/50 px-3 py-2 text-sm text-slate-300">
            {task === "speaker_clone"
              ? "Referans sesler tını klonu için kullanılacak."
              : "Şarkı dönüşümünü MIDI melodisi yönetecek."}
          </div>
        )}
      </div>

      <button
        className="mt-4 w-full rounded-xl bg-gradient-to-r from-cyan-400 to-teal-300 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:from-cyan-300 hover:to-teal-200 disabled:cursor-not-allowed disabled:opacity-60"
        disabled={isBusy}
        onClick={onConvert}
        type="button"
      >
        {isBusy ? "İşleniyor..." : "Dönüşümü Başlat"}
      </button>

      <div className="mt-4 space-y-2 text-sm">
        <p>
          <span className="text-slate-400">Çıktı:</span>{" "}
          <span className="text-slate-200">{outputPath ?? "Henüz sonuç yok"}</span>
        </p>

        <div className="grid gap-2 md:grid-cols-3">
          {Object.keys(metrics).length > 0 ? (
            Object.entries(metrics).map(([key, value]) => (
              <div className="metric-pill" key={key}>
                <div className="text-xs uppercase tracking-wide text-slate-400">{key}</div>
                <div className="mt-1 font-medium text-slate-100">{Number(value).toFixed(3)}</div>
              </div>
            ))
          ) : (
            <div className="text-slate-400">Metrikler dönüşümden sonra görünecek.</div>
          )}
        </div>
      </div>
    </GlassCard>
  );
}
