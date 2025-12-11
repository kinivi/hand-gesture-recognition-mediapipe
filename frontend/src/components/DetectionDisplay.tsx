"use client";

import { motion, AnimatePresence } from "framer-motion";

interface Prediction {
  letter: string;
  confidence: number;
}

interface DetectionDisplayProps {
  letter: string;
  confidence: number;
  topPredictions?: Prediction[];
}

export function DetectionDisplay({ letter, confidence, topPredictions = [] }: DetectionDisplayProps) {
  return (
    <div className="h-full flex flex-col gap-6">
      {/* Current Detection */}
      <div className="rounded-2xl overflow-hidden border border-black/20 dark:border-white/20 bg-white/40 dark:bg-black/40 backdrop-blur-xl shadow-2xl p-8">
        <div className="mb-6">
          <h2>Current Detection</h2>
        </div>

        <AnimatePresence mode="wait">
          {letter ? (
            <motion.div
              key={letter}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="text-center"
            >
              {/* Large Letter Display */}
              <div className="relative mb-6">
                <div className="bg-white/60 dark:bg-black/60 backdrop-blur-xl rounded-2xl p-12 border border-black/20 dark:border-white/20 shadow-lg">
                  <div className="text-9xl tracking-tight font-medium">{letter}</div>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-black/60 dark:text-white/60">Confidence</span>
                  <span className="font-medium">{Math.round(confidence * 100)}%</span>
                </div>
                <div className="h-2 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden backdrop-blur-xl">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${confidence * 100}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                    className="h-full bg-blue-500 rounded-full shadow-lg shadow-blue-500/50"
                  />
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center py-16"
            >
              <div className="text-6xl text-black/20 dark:text-white/20 mb-4">—</div>
              <p className="text-black/40 dark:text-white/40 text-sm">
                Waiting for detection
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Top Predictions */}
      {topPredictions.length > 0 && (
        <div className="rounded-2xl overflow-hidden border border-black/20 dark:border-white/20 bg-white/40 dark:bg-black/40 backdrop-blur-xl shadow-2xl p-6">
          <h3 className="mb-3">Top Predictions</h3>
          <div className="space-y-2">
            {topPredictions.slice(0, 5).map((pred, idx) => (
              <div key={pred.letter} className="flex items-center gap-3">
                <span className={`w-8 h-8 flex items-center justify-center rounded-lg font-medium ${
                  idx === 0 
                    ? "bg-blue-500 text-white" 
                    : "bg-black/10 dark:bg-white/10"
                }`}>
                  {pred.letter}
                </span>
                <div className="flex-1 h-2 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${
                      idx === 0 ? "bg-blue-500" : "bg-black/30 dark:bg-white/30"
                    }`}
                    style={{ width: `${pred.confidence * 100}%` }}
                  />
                </div>
                <span className="text-sm text-black/60 dark:text-white/60 w-12 text-right">
                  {Math.round(pred.confidence * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info Card */}
      <div className="rounded-2xl overflow-hidden border border-black/20 dark:border-white/20 bg-white/40 dark:bg-black/40 backdrop-blur-xl shadow-2xl p-6">
        <h3 className="mb-3">How it works</h3>
        <p className="text-black/60 dark:text-white/60 text-sm leading-relaxed">
          Our real-time detection system uses MediaPipe for hand tracking and custom
          trained neural networks (MLP/LSTM) to recognize ASL hand gestures,
          translating them into letters with high accuracy.
        </p>
      </div>
    </div>
  );
}

