"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import { Hands, Results } from "@mediapipe/hands";

interface Prediction {
  letter: string;
  confidence: number;
}

interface UseASLDetectionProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isActive: boolean;
  onDetection: (letter: string, confidence: number, topPredictions?: Prediction[]) => void;
}

// Default labels (will be overwritten by labels.json if available)
const LABELS: Record<number, string> = {
  0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
  9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q",
  17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 
  25: "Z", 26: "del", 27: "space"
};

const MIN_CONSECUTIVE_SAME = 3;

export function useASLDetection({
  videoRef,
  canvasRef,
  isActive,
  onDetection,
}: UseASLDetectionProps) {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isHandsLoaded, setIsHandsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mlpModelRef = useRef<tf.LayersModel | null>(null);
  const handsRef = useRef<Hands | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const labelsRef = useRef<Record<number, string>>(LABELS);
  
  // Prediction smoothing
  const lastStablePredictionRef = useRef<string>("");
  const consecutiveCountRef = useRef<number>(0);
  const lastRawLetterRef = useRef<string>("");
  const topPredictionsRef = useRef<Prediction[]>([]);

  // Normalize keypoints (same as training - preprocess.py)
  const normalizeKeypoints = useCallback((landmarks: { x: number; y: number; z: number }[]) => {
    const coords = landmarks.map((lm) => [lm.x, lm.y]);
    const baseX = coords[0][0];
    const baseY = coords[0][1];
    const translated = coords.map((c) => [c[0] - baseX, c[1] - baseY]);
    const flat = translated.flatMap((c) => [c[0], c[1]]);
    const maxVal = Math.max(...flat.map(Math.abs));
    
    if (maxVal > 0) {
      return flat.map((v) => v / maxVal);
    }
    return flat;
  }, []);

  // Smooth predictions
  const smoothPrediction = useCallback(
    (letter: string, confidence: number, topPreds?: Prediction[]) => {
      if (topPreds) {
        topPredictionsRef.current = topPreds;
      }

      if (letter === lastRawLetterRef.current) {
        consecutiveCountRef.current++;
      } else {
        consecutiveCountRef.current = 1;
        lastRawLetterRef.current = letter;
      }
      
      const shouldUpdate = 
        consecutiveCountRef.current >= MIN_CONSECUTIVE_SAME ||
        (confidence > 0.7 && letter !== lastStablePredictionRef.current);
      
      if (shouldUpdate && letter !== lastStablePredictionRef.current) {
        lastStablePredictionRef.current = letter;
        onDetection(letter, confidence, topPredictionsRef.current);
      }
    },
    [onDetection]
  );

  // Run MLP inference
  const runMLPInference = useCallback(
    async (keypoints: number[]) => {
      const model = mlpModelRef.current;
      
      if (model) {
        try {
          if (keypoints.length !== 42) {
            console.error(`Wrong keypoints length: ${keypoints.length}, expected 42`);
            return;
          }
          
          const input = tf.tensor2d([keypoints], [1, 42]);
          const prediction = model.predict(input) as tf.Tensor;
          const probabilities = await prediction.data();
          
          const probArray = Array.from(probabilities);
          const indexed = probArray.map((prob, idx) => ({ idx, prob }));
          indexed.sort((a, b) => b.prob - a.prob);
          
          const topPredictions: Prediction[] = indexed.slice(0, 5).map(({ idx, prob }) => ({
            letter: labelsRef.current[idx] || String.fromCharCode(65 + idx),
            confidence: prob,
          }));
          
          smoothPrediction(topPredictions[0].letter, topPredictions[0].confidence, topPredictions);
          
          input.dispose();
          prediction.dispose();
        } catch (e) {
          console.error("MLP inference error:", e);
        }
      }
    },
    [smoothPrediction]
  );

  // Load labels
  useEffect(() => {
    const loadLabels = async () => {
      try {
        const response = await fetch("/models/labels.json");
        if (response.ok) {
          const rawLabels = await response.json();
          const labels: Record<number, string> = {};
          for (const [key, value] of Object.entries(rawLabels)) {
            labels[parseInt(key)] = value as string;
          }
          labelsRef.current = labels;
          console.log("Labels loaded:", Object.keys(labels).length, "classes");
        }
      } catch (e) {
        console.log("Using default labels");
      }
    };
    loadLabels();
  }, []);

  // Load MLP model
  useEffect(() => {
    const loadModel = async () => {
      try {
        setError(null);
        const mlpModel = await tf.loadLayersModel("/models/mlp/model.json");
        mlpModelRef.current = mlpModel;
        console.log("MLP model loaded successfully");
        console.log("Input shape:", mlpModel.inputs[0].shape);
        console.log("Output shape:", mlpModel.outputs[0].shape);
        setIsModelLoaded(true);
      } catch (e) {
        console.error("Failed to load MLP model:", e);
        setError("Failed to load model");
        setIsModelLoaded(true);
      }
    };

    loadModel();

    return () => {
      mlpModelRef.current?.dispose();
    };
  }, []);

  // Initialize MediaPipe Hands
  useEffect(() => {
    const initHands = async () => {
      try {
        const hands = new Hands({
          locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
          },
        });

        hands.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        handsRef.current = hands;
        setIsHandsLoaded(true);
        console.log("MediaPipe Hands initialized");
      } catch (err) {
        console.error("Error initializing MediaPipe Hands:", err);
        setError("Failed to initialize hand detection");
      }
    };

    initHands();

    return () => {
      handsRef.current?.close();
    };
  }, []);

  // Draw hand landmarks
  const drawHandLandmarks = useCallback((
    ctx: CanvasRenderingContext2D,
    landmarks: { x: number; y: number; z: number }[],
    width: number,
    height: number
  ) => {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17],
    ];

    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 3;
    
    for (const [i, j] of connections) {
      ctx.beginPath();
      ctx.moveTo(landmarks[i].x * width, landmarks[i].y * height);
      ctx.lineTo(landmarks[j].x * width, landmarks[j].y * height);
      ctx.stroke();
    }

    ctx.fillStyle = "#22c55e";
    for (const landmark of landmarks) {
      ctx.beginPath();
      ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, []);

  // Clear state when camera stops
  useEffect(() => {
    if (!isActive) {
      lastStablePredictionRef.current = "";
      consecutiveCountRef.current = 0;
      lastRawLetterRef.current = "";
    }
  }, [isActive]);

  // Process video frames
  useEffect(() => {
    if (!isActive || !isHandsLoaded || !videoRef.current || !handsRef.current) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      return;
    }

    const hands = handsRef.current;

    hands.onResults((results: Results) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      
      if (!canvas || !ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        drawHandLandmarks(ctx, landmarks, canvas.width, canvas.height);
        
        const keypoints = normalizeKeypoints(landmarks);
        runMLPInference(keypoints);
      }
    });

    const processFrame = async () => {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        await hands.send({ image: videoRef.current });
      }
      animationFrameRef.current = requestAnimationFrame(processFrame);
    };

    processFrame();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isActive, isHandsLoaded, videoRef, canvasRef, normalizeKeypoints, drawHandLandmarks, runMLPInference]);

  return {
    isModelLoaded,
    isHandsLoaded,
    error,
    isReady: isModelLoaded && isHandsLoaded,
  };
}
