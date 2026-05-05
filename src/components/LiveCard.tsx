import { GlassCard } from "@/components/GlassCard";

interface LiveCardProps {
  isLive: boolean;
  isBusy: boolean;
  routeToVirtualMic: boolean;
  virtualMicDevices: string[];
  selectedVirtualMic: string | null;
  activityLevel: number;
  onToggleLive: () => void;
  onRefreshDevices: () => void;
  onRouteChange: (enabled: boolean) => void;
  onVirtualMicChange: (device: string | null) => void;
}

export function LiveCard({
  isLive,
  isBusy,
  routeToVirtualMic,
  virtualMicDevices,
  selectedVirtualMic,
  activityLevel,
  onToggleLive,
  onRefreshDevices,
  onRouteChange,
  onVirtualMicChange,
}: LiveCardProps) {
  return (
    <GlassCard title="Live Microphone" subtitle="Capture mic audio and stream transformed voice in real time.">
      <div className="space-y-3">
        <button
          className="w-full rounded-xl border border-cyan-300/40 bg-cyan-500/20 px-4 py-2.5 text-sm font-semibold transition hover:bg-cyan-400/30 disabled:opacity-60"
          disabled={isBusy}
          onClick={onToggleLive}
          type="button"
        >
          {isLive ? "Stop Live Session" : "Start Live Session"}
        </button>

        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            checked={routeToVirtualMic}
            className="h-4 w-4 rounded border-slate-400/40 bg-slate-900"
            onChange={(event) => onRouteChange(event.target.checked)}
            type="checkbox"
          />
          Route processed audio to virtual microphone
        </label>

        <div className="grid grid-cols-[1fr_auto] gap-2">
          <select
            className="rounded-xl border border-slate-300/20 bg-slate-900/70 px-3 py-2 text-sm"
            onChange={(event) => onVirtualMicChange(event.target.value || null)}
            value={selectedVirtualMic ?? ""}
          >
            <option value="">System default output</option>
            {virtualMicDevices.map((device) => (
              <option key={device} value={device}>
                {device}
              </option>
            ))}
          </select>

          <button
            className="rounded-xl border border-slate-300/20 bg-slate-800/70 px-3 py-2 text-sm hover:border-slate-200/30"
            onClick={onRefreshDevices}
            type="button"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="mt-4">
        <div className="mb-2 text-xs uppercase tracking-wide text-slate-400">Live activity</div>
        <div className="h-3 overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-emerald-300 to-lime-300 transition-all"
            style={{ width: `${Math.max(4, Math.min(activityLevel * 100, 100))}%` }}
          />
        </div>
      </div>
    </GlassCard>
  );
}
