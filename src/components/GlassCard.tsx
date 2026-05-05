import React from "react";

interface GlassCardProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}

export function GlassCard({ title, subtitle, children, className }: GlassCardProps) {
  return (
    <section className={`glass-card p-5 md:p-6 ${className ?? ""}`}>
      <header className="mb-4">
        <h2 className="font-display text-xl font-semibold tracking-tight text-white">{title}</h2>
        {subtitle ? <p className="mt-1 text-sm text-slate-300/80">{subtitle}</p> : null}
      </header>
      {children}
    </section>
  );
}
