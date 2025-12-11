import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      animation: {
        "spotlight": "spotlight 15s ease-in-out infinite",
        "spotlight-centered": "spotlight-centered 20s ease-in-out infinite",
        "spotlight-bg": "spotlight-bg 25s ease-in-out infinite",
        "spotlight-bg-2": "spotlight-bg-2 30s ease-in-out infinite",
      },
      keyframes: {
        spotlight: {
          "0%, 100%": { transform: "translate(-100px, -100px)" },
          "50%": { transform: "translate(calc(100% - 400px), calc(100% - 400px))" },
        },
        "spotlight-centered": {
          "0%": { transform: "translate(-10%, -10%) scale(1)" },
          "25%": { transform: "translate(5%, -15%) scale(1.1)" },
          "50%": { transform: "translate(10%, -5%) scale(0.95)" },
          "75%": { transform: "translate(-5%, 10%) scale(1.05)" },
          "100%": { transform: "translate(-10%, -10%) scale(1)" },
        },
        "spotlight-bg": {
          "0%": { transform: "translate(-10%, -10%)" },
          "25%": { transform: "translate(60%, 20%)" },
          "50%": { transform: "translate(80%, 50%)" },
          "75%": { transform: "translate(20%, 60%)" },
          "100%": { transform: "translate(-10%, -10%)" },
        },
        "spotlight-bg-2": {
          "0%": { transform: "translate(70%, 40%)" },
          "25%": { transform: "translate(10%, 50%)" },
          "50%": { transform: "translate(-5%, 10%)" },
          "75%": { transform: "translate(50%, -5%)" },
          "100%": { transform: "translate(70%, 40%)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;

