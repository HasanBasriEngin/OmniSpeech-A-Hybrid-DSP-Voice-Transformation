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
    <GlassCard title="Audio Source" subtitle="Load voice files for conversion and optional references.">
      <div className="space-y-3">
        <button
          className="w-full rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-teal-300"
          onClick={onPickSource}
          type="button"
        >
          Select Input Audio
        </button>

        <button
          className="w-full rounded-xl border border-slate-300/20 bg-slate-800/70 px-4 py-2.5 text-sm font-medium transition hover:border-slate-200/30"
          onClick={onPickReferences}
          type="button"
        >
          Select Speaker References
        </button>

        <button
          className="w-full rounded-xl border border-slate-300/20 bg-slate-800/70 px-4 py-2.5 text-sm font-medium transition hover:border-slate-200/30"
          onClick={onPickMidi}
          type="button"
        >
          Select MIDI (Singing)
        </button>
      </div>

      <div className="mt-4 space-y-2 text-sm text-slate-200/90">
        <p>
          <span className="text-slate-400">Input:</span>{" "}
          {sourceFile ? basename(sourceFile) : "Not selected"}
        </p>
        <p>
          <span className="text-slate-400">References:</span>{" "}
          {referenceFiles.length > 0
            ? `${referenceFiles.length} file(s)`
            : "Not selected"}
        </p>
        <p>
          <span className="text-slate-400">MIDI:</span>{" "}
          {midiFile ? basename(midiFile) : "Not selected"}
        </p>
      </div>
    </GlassCard>
  );
}
