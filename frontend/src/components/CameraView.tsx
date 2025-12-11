"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Loader2 } from "lucide-react";
import { useASLDetection } from "@/hooks/useASLDetection";

interface Prediction {
  letter: string;
  confidence: number;
}

interface CameraViewProps {
  onDetection: (letter: string, confidence: number, topPredictions?: Prediction[]) => void;
}

export function CameraView({ onDetection }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [error, setError] = useState<string>("");

  const { isReady, error: detectionError } = useASLDetection({
    videoRef,
    canvasRef,
    isActive: isCameraActive,
    onDetection,
  });

  // Update canvas size when video loads
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (video && canvas) {
      const updateSize = () => {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
      };
      
      video.addEventListener("loadedmetadata", updateSize);
      return () => video.removeEventListener("loadedmetadata", updateSize);
    }
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
        setError("");
      }
    } catch (err) {
      setError("Camera access denied. Please enable camera permissions.");
      console.error("Camera error:", err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    }
  };

  return (
    <div className="relative rounded-2xl overflow-hidden border border-black/20 dark:border-white/20 backdrop-blur-xl bg-white/40 dark:bg-black/40 shadow-2xl">
      {/* Animated Blue Spotlight Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none flex items-center justify-center">
        <div className="absolute w-[500px] h-[500px] bg-blue-500/35 dark:bg-blue-500/30 rounded-full blur-[120px] animate-spotlight-centered" />
      </div>

      {/* Video Container */}
      <div className="relative aspect-video bg-black/5 dark:bg-white/5 backdrop-blur-sm">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover relative z-10"
          style={{ transform: "scaleX(-1)" }}
        />
        
        {/* Canvas overlay for hand landmarks */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full z-20 pointer-events-none"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Overlay when camera is off */}
        {!isCameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/90 dark:bg-black/90 backdrop-blur-xl z-30">
            <CameraOff className="w-12 h-12 text-black/30 dark:text-white/30 mb-4" />
            <p className="text-black/60 dark:text-white/60">Camera is currently off</p>
            {!isReady && (
              <div className="flex items-center gap-2 mt-4 text-black/40 dark:text-white/40">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm">Loading model...</span>
              </div>
            )}
          </div>
        )}

        {/* Error Message */}
        {(error || detectionError) && (
          <div className="absolute top-4 left-4 right-4 bg-white/90 dark:bg-black/90 backdrop-blur-xl border border-black/20 dark:border-white/20 rounded-xl p-4 z-40 shadow-lg">
            <p className="text-sm">{error || detectionError}</p>
          </div>
        )}

        {/* Detection Indicator */}
        {isCameraActive && (
          <div className="absolute top-4 right-4 z-40">
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/90 dark:bg-black/90 backdrop-blur-xl border border-blue-500/50 shadow-lg">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              <span className="text-sm">Detecting</span>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="p-6 border-t border-black/20 dark:border-white/20 relative z-10 bg-white/60 dark:bg-black/60 backdrop-blur-xl">
        <div className="flex items-center justify-between">
          <div>
            <p className="mb-1 font-medium">Camera Feed</p>
            <p className="text-black/60 dark:text-white/60 text-sm">
              {isCameraActive
                ? "Real-time ASL detection active"
                : "Start camera to begin detection"}
            </p>
          </div>

          <button
            onClick={isCameraActive ? stopCamera : startCamera}
            disabled={!isReady && !isCameraActive}
            className="flex items-center gap-2 px-6 py-3 rounded-full border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60 backdrop-blur-xl hover:bg-white/80 dark:hover:bg-black/80 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isCameraActive ? (
              <>
                <CameraOff className="w-4 h-4" />
                Stop
              </>
            ) : (
              <>
                <Camera className="w-4 h-4" />
                {isReady ? "Start" : "Loading..."}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
