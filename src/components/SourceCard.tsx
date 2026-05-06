import { GlassCard } from "@/components/GlassCard";

interface SourceCardProps {
  sourceFile: string | null;
  referenceFiles: string[];
  midiFile: string | null;
  onPickSource: () => void;
  onPickReferences: () => void;
  onPickMidi: () => void;
}

const basename = (path: string) => path.split(/[\\/]/).pop() ?? path;

export function SourceCard({
  sourceFile,
  referenceFiles,
  midiFile,
  onPickSource,
  onPickReferences,
  onPickMidi,
}: SourceCardProps) {
  return (
    <GlassCard title="Ses Kaynağı" subtitle="Dönüşüm için ses dosyası ve isteğe bağlı referansları yükle.">
      <div className="space-y-3">
        <button
          className="w-full rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-teal-300"
          onClick={onPickSource}
          type="button"
        >
          Kaynak Ses Seç
        </button>

        <button
          className="w-full rounded-xl border border-slate-300/20 bg-slate-800/70 px-4 py-2.5 text-sm font-medium transition hover:border-slate-200/30"
          onClick={onPickReferences}
          type="button"
        >
          Konuşmacı Referansı Seç
        </button>

        <button
          className="w-full rounded-xl border border-slate-300/20 bg-slate-800/70 px-4 py-2.5 text-sm font-medium transition hover:border-slate-200/30"
          onClick={onPickMidi}
          type="button"
        >
          MIDI Seç (Şarkı)
        </button>
      </div>

      <div className="mt-4 space-y-2 text-sm text-slate-200/90">
        <p>
          <span className="text-slate-400">Kaynak:</span>{" "}
          {sourceFile ? basename(sourceFile) : "Seçilmedi"}
        </p>
        <p>
          <span className="text-slate-400">Referanslar:</span>{" "}
          {referenceFiles.length > 0
            ? `${referenceFiles.length} dosya`
            : "Seçilmedi"}
        </p>
        <p>
          <span className="text-slate-400">MIDI:</span>{" "}
          {midiFile ? basename(midiFile) : "Seçilmedi"}
        </p>
      </div>
    </GlassCard>
  );
}
