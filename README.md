# GestureEdge — Hand Gesture Control on Edge Hardware

[🇻🇳 Tiếng Việt](README_VI.md)

Control the Chrome Dino game with hand gestures. Three gesture classes. Inference at 0.038ms. Gesture recognition runs in a background thread, independent from the game loop.

---

## Demo

<!-- TODO: add demo GIF after recording -->

| Gesture | Action |
|---------|--------|
| Open hand | Jump |
| Fist | Duck |
| Pointer finger | Run |

---

## System Pipeline

> 👆 Click the pipeline image to explore each stage in detail.

<a href="https://ChuoiUhuhu0727.github.io/gesture-edge-ai/pipeline.html">
  <img src="pipeline.svg" width="280" alt="System Pipeline"/>
</a>

*Gesture recognition runs in a daemon thread. The game runs in the main thread. They share one variable protected by `threading.Lock()`.*

---

## Engineering Decisions

### 1. Landmark normalization

Raw pixel coordinates change with hand position and camera distance. I normalize each frame:
1. Shift all 21 points relative to the wrist (landmark 0)
2. Scale by the largest absolute value in the 42-dim vector

The MLP then works for any hand size and camera distance without retraining.

### 2. Threading

Gesture inference blocks on I/O and model execution. Running it inside the game loop would freeze the game on slow frames.

Fix: inference in a daemon thread, game in the main thread. They share one `_gesture_action` variable behind a `Lock`. The game reads it each frame with zero wait.

### 3. Temporal smoothing — GestureBuffer

One bad frame (finger partly hidden) can flip the class and trigger a random jump. `GestureBuffer` applies a sliding window majority vote over 10 frames to smooth this out.

It also holds the last stable gesture for 5 frames after detection drops — so brief occlusion doesn't reset the action.

```
raw frame result (or -1 = no hand)
      ↓
GestureBuffer(window=10, hold_frames=5)
      ↓
stable gesture → game key
```

### 4. Why ONNX, not TensorRT

I benchmarked both before choosing:

| Runtime | Latency |
|---------|---------|
| ONNX CPU | **0.038 ms** |
| TensorRT GPU | 0.062 ms |

TensorRT was 63% slower. The model has only 1,114 parameters — too small for GPU parallelism to pay off. Memory transfer overhead (CPU↔GPU) is larger than the actual computation. TensorRT wins with heavy models (CNNs, transformers). Not here.

### 5. Glove robustness

Thick gloves caused intermittent detection — landmarks appeared and disappeared randomly, making the dino jump uncontrollably.

**Hypothesis:** MediaPipe was trained on bare hands. Gloves change the hand's silhouette and hide the joint creases the detector relies on.

What I applied:
- Lowered `min_hand_detection_confidence` to 0.4 (from default 0.5)
- `hold_frames=5` in GestureBuffer to survive brief detection gaps

This improved stability but did not fully fix it. A complete fix requires retraining the palm detector on gloved-hand data — outside this project's scope.

---

## Results

| Metric | Value |
|--------|-------|
| ONNX CPU latency | 0.038 ms |
| TensorRT GPU latency | 0.062 ms |
| Gesture classes | 3 active / 4 trained |
| Classifier accuracy | 97.2% |
| Smoothing window | 10 frames |
| Hold on miss | 5 frames |

---

## Stack

| Layer | Tech |
|-------|------|
| Hand detection | MediaPipe Tasks |
| Classifier | TFLite MLP (42→20→10→4, 1,114 params) |
| Camera | OpenCV |
| Threading | Python `threading` |
| Runtime | ONNX Runtime (CPU) |
| Game | Chrome Dino Runner (monkey-patched) |

---

## Setup

```bash
pip install mediapipe opencv-python numpy
```

Clone Chrome Dino Runner (external dependency):

```bash
git clone https://github.com/dhhruv/Chrome-Dino-Runner.git Chrome-Dino-Runner-master
```

Run:

```bash
python gesture_dino.py
```

Keys still work: `Up`/`Space` = jump, `Down` = duck, `p` = pause, `ESC` = close camera.

---

## Credits

- Gesture recognition base: [Kazuhito00](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) (translated by [kinivi](https://github.com/kinivi))
- Game: [dhhruv/Chrome-Dino-Runner](https://github.com/dhhruv/Chrome-Dino-Runner)
