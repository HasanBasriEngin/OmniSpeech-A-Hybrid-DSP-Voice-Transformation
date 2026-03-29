import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Space Grotesk", "Segoe UI", "sans-serif"],
        body: ["IBM Plex Sans", "Segoe UI", "sans-serif"],
      },
      colors: {
        backdrop: "#080B14",
        panel: "#111826",
        accent: "#0ea5a6",
        accentSoft: "#14b8a61a",
      },
      boxShadow: {
        glass: "0 14px 38px rgba(6, 10, 25, 0.45)",
      },
      animation: {
        drift: "drift 14s ease-in-out infinite",
      },
      keyframes: {
        drift: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
