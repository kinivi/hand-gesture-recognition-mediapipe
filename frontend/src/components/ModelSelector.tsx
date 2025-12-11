"use client";

import { Brain } from "lucide-react";

export function ModelSelector() {
  return (
    <div className="flex items-center gap-2 p-2 rounded-full border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60 backdrop-blur-xl">
      <div className="flex items-center gap-2 px-4 py-2 rounded-full text-sm bg-blue-500 text-white shadow-lg shadow-blue-500/30">
        <Brain className="w-4 h-4" />
        MLP Model
      </div>
    </div>
  );
}
