"use client";

import { useState } from "react";
import { CameraView } from "@/components/CameraView";
import { DetectionDisplay } from "@/components/DetectionDisplay";
import { ModelSelector } from "@/components/ModelSelector";
import { ResearchPaper } from "@/components/ResearchPaper";
import { ThemeToggle } from "@/components/ThemeToggle";
import { useTheme } from "@/hooks/useTheme";

interface Prediction {
  letter: string;
  confidence: number;
}

export default function Home() {
  const [detectedLetter, setDetectedLetter] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [topPredictions, setTopPredictions] = useState<Prediction[]>([]);
  const { theme, toggleTheme } = useTheme();

  return (
    <div className={theme === "dark" ? "dark" : ""}>
      <div className="min-h-screen bg-white dark:bg-black text-black dark:text-white transition-colors duration-300">
        {/* Header */}
        <header className="border-b border-black/20 dark:border-white/20 backdrop-blur-xl bg-white/80 dark:bg-black/80 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
            <div>
              <h1 className="tracking-tight">ASL Live Translator</h1>
              <p className="text-black/60 dark:text-white/60 mt-1">
                Real-time American Sign Language Recognition
              </p>
            </div>
            <div className="flex items-center gap-4">
              <ModelSelector />
              <ThemeToggle theme={theme} onToggle={toggleTheme} />
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-6 py-8 relative z-10">
          {/* Camera and Detection Section */}
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-20 relative">
            {/* Animated Background Spotlights */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden">
              <div className="absolute w-[700px] h-[700px] bg-blue-500/40 dark:bg-blue-500/50 rounded-full blur-[130px] animate-spotlight-bg" />
              <div className="absolute w-[500px] h-[500px] bg-blue-400/30 dark:bg-blue-400/40 rounded-full blur-[100px] animate-spotlight-bg-2" />
            </div>

            {/* Camera View - Takes 3 columns */}
            <div className="lg:col-span-3 relative z-10">
              <CameraView
                onDetection={(letter, conf, preds) => {
                  setDetectedLetter(letter);
                  setConfidence(conf);
                  if (preds) setTopPredictions(preds);
                }}
              />
            </div>

            {/* Detection Display - Takes 1 column */}
            <div className="lg:col-span-1 relative z-10">
              <DetectionDisplay 
                letter={detectedLetter} 
                confidence={confidence}
                topPredictions={topPredictions}
              />
            </div>
          </div>

          {/* Research Paper Section */}
          <ResearchPaper />
        </main>
      </div>
    </div>
  );
}
