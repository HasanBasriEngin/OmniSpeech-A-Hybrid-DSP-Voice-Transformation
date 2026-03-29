import { GlassCard } from "@/components/GlassCard";

interface LogCardProps {
  logs: string[];
}

export function LogCard({ logs }: LogCardProps) {
  return (
    <GlassCard title="Pipeline Logs" subtitle="Tauri IPC + Python backend activity stream.">
      <div className="max-h-56 space-y-1 overflow-auto rounded-xl border border-slate-200/10 bg-slate-950/60 p-3 text-xs text-slate-300">
        {logs.length === 0 ? <p>No logs yet.</p> : null}
        {logs.map((entry, index) => (
          <p key={`${entry}-${index}`}>{entry}</p>
        ))}
      </div>
    </GlassCard>
  );
}
