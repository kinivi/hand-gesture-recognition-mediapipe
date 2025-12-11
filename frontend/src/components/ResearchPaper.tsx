"use client";

import { Brain, Layers, Zap, CheckCircle, AlertTriangle, Users, Code, Lightbulb } from "lucide-react";

export function ResearchPaper() {
  return (
    <div className="rounded-2xl overflow-hidden border border-black/20 dark:border-white/20 bg-white/40 dark:bg-black/40 backdrop-blur-xl shadow-2xl">
      <div className="p-12 max-w-4xl mx-auto">
        {/* Title Section */}
        <div className="text-center mb-12 pb-8 border-b border-black/20 dark:border-white/20">
          <h1 className="mb-4 text-2xl font-medium">
            Real-Time ASL Recognition Using Neural Networks
          </h1>
          <p className="text-black/60 dark:text-white/60">
            ECS 170 - Introduction to Artificial Intelligence
          </p>
          <div className="flex items-center justify-center gap-6 mt-6 text-sm text-black/50 dark:text-white/50">
            <span>Group Project</span>
            <span>•</span>
            <span>Fall 2025</span>
          </div>
        </div>

        {/* Project Overview */}
        <section className="mb-12">
          <div className="inline-block px-4 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 backdrop-blur-xl mb-4">
            <span className="text-blue-600 dark:text-blue-400 text-sm">Project Overview</span>
          </div>
          <p className="text-black/70 dark:text-white/70 leading-relaxed mb-4">
            This web application demonstrates real-time American Sign Language (ASL) alphabet 
            recognition using deep learning and computer vision. The system captures video from 
            your webcam, detects hand landmarks using MediaPipe, and classifies the hand gesture 
            into one of 26 ASL letters (A-Z) using custom-trained neural networks.
          </p>
          <p className="text-black/70 dark:text-white/70 leading-relaxed">
            The entire inference pipeline runs directly in your browser using TensorFlow.js, 
            requiring no server-side processing. This enables real-time predictions without 
            any backend server, making deployment simple and accessible.
          </p>
        </section>

        {/* System Architecture */}
        <section className="mb-12">
          <h2 className="mb-6 text-xl font-medium">System Architecture</h2>

          {/* Architecture Diagram */}
          <div className="mb-8 p-8 rounded-xl bg-white/60 dark:bg-black/60 backdrop-blur-xl border border-black/20 dark:border-white/20 shadow-lg">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <div className="flex-1 min-w-[120px]">
                <div className="p-4 rounded-lg border border-black/20 dark:border-white/20 bg-white/80 dark:bg-black/80 backdrop-blur-xl text-center shadow-lg">
                  <Code className="w-6 h-6 text-black/60 dark:text-white/60 mx-auto mb-2" />
                  <div className="text-sm mb-1 font-medium">Webcam</div>
                  <div className="text-black/50 dark:text-white/50 text-xs">Video Input</div>
                </div>
              </div>

              <div className="text-black/40 dark:text-white/40">→</div>

              <div className="flex-1 min-w-[120px]">
                <div className="p-4 rounded-lg border border-black/20 dark:border-white/20 bg-white/80 dark:bg-black/80 backdrop-blur-xl text-center shadow-lg">
                  <Layers className="w-6 h-6 text-black/60 dark:text-white/60 mx-auto mb-2" />
                  <div className="text-sm mb-1 font-medium">MediaPipe</div>
                  <div className="text-black/50 dark:text-white/50 text-xs">21 Landmarks</div>
                </div>
              </div>

              <div className="text-black/40 dark:text-white/40">→</div>

              <div className="flex-1 min-w-[120px]">
                <div className="p-4 rounded-lg border border-black/20 dark:border-white/20 bg-white/80 dark:bg-black/80 backdrop-blur-xl text-center shadow-lg">
                  <Zap className="w-6 h-6 text-black/60 dark:text-white/60 mx-auto mb-2" />
                  <div className="text-sm mb-1 font-medium">Normalize</div>
                  <div className="text-black/50 dark:text-white/50 text-xs">42 Features</div>
                </div>
              </div>

              <div className="text-black/40 dark:text-white/40">→</div>

              <div className="flex-1 min-w-[120px]">
                <div className="p-4 rounded-lg border border-blue-500/30 dark:border-blue-500/30 bg-blue-500/10 dark:bg-blue-500/20 backdrop-blur-xl text-center shadow-lg">
                  <Brain className="w-6 h-6 text-blue-600 dark:text-blue-400 mx-auto mb-2" />
                  <div className="text-sm mb-1 font-medium">MLP/LSTM</div>
                  <div className="text-black/50 dark:text-white/50 text-xs">TensorFlow.js</div>
                </div>
              </div>
            </div>
            <p className="text-center text-black/50 dark:text-white/50 text-sm mt-6">
              Figure 1: End-to-end inference pipeline running entirely in the browser
            </p>
          </div>
        </section>

        {/* AI Methodologies */}
        <section className="mb-12">
          <h2 className="mb-6 text-xl font-medium">AI Methodologies & Techniques</h2>

          <div className="space-y-6">
            {/* Data Processing */}
            <div>
              <h3 className="mb-3 text-lg font-medium flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-blue-500" />
                Hand Landmark Detection
              </h3>
              <p className="text-black/70 dark:text-white/70 leading-relaxed mb-3">
                We use Google&apos;s MediaPipe Hands to detect 21 3D landmarks on each hand in real-time. 
                Each landmark represents a joint or fingertip position (x, y coordinates normalized to 0-1).
              </p>
              <div className="p-4 rounded-lg bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10">
                <code className="text-sm text-black/70 dark:text-white/70">
                  Landmarks: Wrist, Thumb (4), Index (4), Middle (4), Ring (4), Pinky (4) = 21 points
                </code>
              </div>
            </div>

            {/* Feature Engineering */}
            <div>
              <h3 className="mb-3 text-lg font-medium flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-blue-500" />
                Feature Engineering
              </h3>
              <p className="text-black/70 dark:text-white/70 leading-relaxed mb-3">
                Raw landmark coordinates are transformed into a normalized feature vector:
              </p>
              <ul className="space-y-2 text-black/70 dark:text-white/70">
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 mt-1">1.</span>
                  <span><strong>Translation invariance:</strong> All coordinates are relative to the wrist position</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 mt-1">2.</span>
                  <span><strong>Scale invariance:</strong> Normalized by the maximum absolute coordinate value</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-500 mt-1">3.</span>
                  <span><strong>2D only:</strong> We use only X,Y coordinates (not Z depth) for stability</span>
                </li>
              </ul>
              <p className="text-black/60 dark:text-white/60 text-sm mt-3">
                Final feature vector: 21 landmarks × 2 coordinates = <strong>42 features</strong>
              </p>
            </div>

            {/* Models */}
            <div>
              <h3 className="mb-3 text-lg font-medium flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-blue-500" />
                Neural Network Models
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="p-5 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60 backdrop-blur-xl shadow-lg">
                  <div className="font-medium mb-2 flex items-center gap-2">
                    <Brain className="w-4 h-4 text-blue-500" />
                    Multi-Layer Perceptron (MLP)
                  </div>
                  <div className="text-black/60 dark:text-white/60 text-sm space-y-1">
                    <p>• Input: 42 features</p>
                    <p>• Hidden: 256 → 128 → 64 neurons</p>
                    <p>• Activation: ReLU + BatchNorm + Dropout</p>
                    <p>• Output: 28 classes (A-Z + special)</p>
                    <p className="text-blue-600 dark:text-blue-400 mt-2">Best for: Static hand poses</p>
                  </div>
                </div>
                
                <div className="p-5 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60 backdrop-blur-xl shadow-lg">
                  <div className="font-medium mb-2 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-blue-500" />
                    Long Short-Term Memory (LSTM)
                  </div>
                  <div className="text-black/60 dark:text-white/60 text-sm space-y-1">
                    <p>• Input: 30 frames × 42 features</p>
                    <p>• Bidirectional LSTM: 128 → 64 units</p>
                    <p>• Temporal context: ~1 second window</p>
                    <p>• Output: 28 classes</p>
                    <p className="text-blue-600 dark:text-blue-400 mt-2">Best for: Dynamic gestures (J, Z)</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Training */}
            <div>
              <h3 className="mb-3 text-lg font-medium flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-blue-500" />
                Training Process
              </h3>
              <p className="text-black/70 dark:text-white/70 leading-relaxed mb-3">
                Models were trained on the Kaggle ASL Alphabet dataset:
              </p>
              <ul className="space-y-2 text-black/70 dark:text-white/70">
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>Images across 29 classes (A-Z, space, delete, nothing)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>MediaPipe extracted keypoints from each image</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>80/10/10 train/validation/test split with stratification</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></span>
                  <span>Early stopping and learning rate scheduling to prevent overfitting</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Challenges */}
        <section className="mb-12">
          <h2 className="mb-6 text-xl font-medium flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            Challenges & Solutions
          </h2>

          <div className="space-y-6">
            <div className="p-5 rounded-xl border border-amber-500/30 bg-amber-500/5 dark:bg-amber-500/10">
              <h4 className="font-medium mb-2">Challenge: Prediction Flickering</h4>
              <p className="text-black/70 dark:text-white/70 text-sm mb-2">
                Raw model predictions changed rapidly frame-to-frame, causing the displayed letter 
                to flicker even when holding a steady pose.
              </p>
              <div className="flex items-start gap-2">
                <Lightbulb className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <p className="text-green-700 dark:text-green-400 text-sm">
                  <strong>Solution:</strong> Implemented temporal smoothing with consecutive-frame voting. 
                  The display only updates after seeing the same prediction consistently across multiple frames.
                </p>
              </div>
            </div>

            <div className="p-5 rounded-xl border border-amber-500/30 bg-amber-500/5 dark:bg-amber-500/10">
              <h4 className="font-medium mb-2">Challenge: Training vs. Inference Mismatch</h4>
              <p className="text-black/70 dark:text-white/70 text-sm mb-2">
                The model trained on static Kaggle images performed poorly on live webcam input. 
                The Z-depth coordinate from MediaPipe was particularly unstable.
              </p>
              <div className="flex items-start gap-2">
                <Lightbulb className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <p className="text-green-700 dark:text-green-400 text-sm">
                  <strong>Solution:</strong> Switched to 2D-only features (X,Y without Z), matching the 
                  approach used in proven hand gesture recognition systems. This improved stability significantly.
                </p>
              </div>
            </div>

            <div className="p-5 rounded-xl border border-amber-500/30 bg-amber-500/5 dark:bg-amber-500/10">
              <h4 className="font-medium mb-2">Challenge: Browser Deployment</h4>
              <p className="text-black/70 dark:text-white/70 text-sm mb-2">
                Running ML models in the browser while maintaining real-time performance required careful optimization.
              </p>
              <div className="flex items-start gap-2">
                <Lightbulb className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <p className="text-green-700 dark:text-green-400 text-sm">
                  <strong>Solution:</strong> Used TensorFlow.js for model inference and MediaPipe&apos;s 
                  WebAssembly-accelerated hand detection. Models were converted from Keras to TFJS format.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Team Contributions */}
        <section className="mb-8 pt-8 border-t border-black/20 dark:border-white/20">
          <h2 className="mb-6 text-xl font-medium flex items-center gap-2">
            <Users className="w-5 h-5 text-blue-500" />
            Team Contributions
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Caitlyn Ho</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Frontend Development, UI/UX Design, Deployment, Model Training
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Mohamed Abdullrahma</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Data Preprocessing, Feature Engineering, Model Architecture
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">David Bui</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Backend Integration, Testing
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Teddy Liu</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Model Training, Data Collection
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">David Arista</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Dataset Preparation, Research
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Johnny Betmansour</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Data Processing, Testing
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Carlos Rayo</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Documentation, Research
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Xiyan Zeng</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Data Augmentation, Model Export
              </p>
            </div>
            <div className="p-4 rounded-xl border border-black/20 dark:border-white/20 bg-white/60 dark:bg-black/60">
              <div className="font-medium mb-2">Norman Gutierrez-Ugalde</div>
              <p className="text-black/60 dark:text-white/60 text-sm">
                Quality Assurance, Documentation
              </p>
            </div>
          </div>
        </section>

        {/* References */}
        <section className="pt-8 border-t border-black/20 dark:border-white/20">
          <h2 className="mb-4 text-xl font-medium">References</h2>
          <ol className="space-y-3 text-sm text-black/60 dark:text-white/60">
            <li>[1] Zhang, F., et al. &quot;MediaPipe Hands: On-device Real-time Hand Tracking.&quot; CVPR Workshop 2020.</li>
            <li>[2] Kaggle ASL Alphabet Dataset. https://www.kaggle.com/datasets/grassknoted/asl-alphabet</li>
            <li>[3] TensorFlow.js Documentation. https://www.tensorflow.org/js</li>
          </ol>
        </section>
      </div>
    </div>
  );
}
