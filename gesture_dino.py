"""
gesture_dino.py  (v2)
Run: python gesture_dino.py

Requires Chrome-Dino-Runner-master/ folder in the same directory.
Clone it from: https://github.com/shivamkapasia0/Chrome-Dino-Runner

Gesture controls
  Open hand  (id 0)  ->  Jump
  Fist       (id 1)  ->  Duck
  Pointer    (id 2)  ->  Run normally  (cancel jump/duck)
  Any gesture        ->  Start game from menu

Keyboard still works:  Up/Space = jump,  Down = duck,  p = pause,  u = unpause
ESC in camera window closes camera (game keeps running with keyboard).
"""

import sys, os, threading, copy, itertools
from collections import deque, Counter

# ── paths ──────────────────────────────────────────────────────────────────────
GESTURE_DIR = os.path.dirname(os.path.abspath(__file__))
DINO_DIR    = os.path.normpath(
    os.path.join(GESTURE_DIR, "Chrome-Dino-Runner-master", "Chrome-Dino-Runner-master")
)
os.chdir(DINO_DIR)          # game loads assets with relative paths
sys.path.insert(0, GESTURE_DIR)
sys.path.insert(0, DINO_DIR)

# ── imports ────────────────────────────────────────────────────────────────────
import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from chromedino import DinoGame

# ── GestureBuffer: temporal smoothing via majority vote ───────────────────────
class GestureBuffer:
    """Sliding window majority vote over the last `window` frames.

    Also holds the last stable gesture for `hold_frames` after detection drops,
    so brief misses (e.g. glove occlusion) don't reset the action mid-game.
    """

    def __init__(self, window=10, hold_frames=5):
        self._buf          = deque(maxlen=window)
        self._hold_frames  = hold_frames
        self._miss_count   = 0

    def update(self, raw_id):
        """raw_id: int gesture class, or -1 when no hand detected."""
        if raw_id == -1:
            self._miss_count += 1
            if self._miss_count <= self._hold_frames:
                pass            # hold: don't push -1, keep old votes intact
            else:
                self._buf.append(-1)    # truly gone → let -1 votes build up
        else:
            self._miss_count = 0
            self._buf.append(raw_id)

        if not self._buf:
            return -1

        winner, _ = Counter(self._buf).most_common(1)[0]
        return winner


# ── shared gesture state ───────────────────────────────────────────────────────
_gesture_lock   = threading.Lock()
_gesture_action = None        # 'jump' | 'duck' | 'run' | None

def _set_gesture(action):
    global _gesture_action
    with _gesture_lock:
        _gesture_action = action

def _get_gesture():
    with _gesture_lock:
        return _gesture_action

# ── landmark helpers ───────────────────────────────────────────────────────────
def _calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in landmarks
    ]

def _pre_process_landmark(landmark_list):
    tmp = copy.deepcopy(landmark_list)
    base_x, base_y = tmp[0]
    for pt in tmp:
        pt[0] -= base_x
        pt[1] -= base_y
    flat  = list(itertools.chain.from_iterable(tmp))
    max_v = max(map(abs, flat)) or 1
    return [v / max_v for v in flat]

# ── gesture → action mapping ───────────────────────────────────────────────────
# Index must match keypoint_classifier_label.csv row order:
#   0 = Open, 1 = Close, 2 = Pointer, 3 = OK
GESTURE_JUMP = 0
GESTURE_DUCK = 1
GESTURE_RUN  = 2   # Pointer: explicitly cancel jump/duck and run

# (action, display text, BGR colour)
_LABEL_MAP = {
    GESTURE_JUMP: ('jump', 'JUMP  (open hand)',  (0,  210,  0)),
    GESTURE_DUCK: ('duck', 'DUCK  (fist)',        (0,   50, 255)),
    GESTURE_RUN:  ('run',  'RUN   (pointer)',     (255, 170,  0)),
}

# ── gesture recognition thread ─────────────────────────────────────────────────
def _gesture_thread():
    model_path = os.path.join(GESTURE_DIR, "hand_landmarker.task")
    base_opts  = mp.tasks.BaseOptions(model_asset_path=model_path)
    options    = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.4,   # lowered: helps with gloves
        min_tracking_confidence=0.3,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    kp_path    = os.path.join(GESTURE_DIR, "model", "keypoint_classifier",
                              "keypoint_classifier.tflite")
    classifier = KeyPointClassifier(model_path=kp_path)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("[gesture] Cannot open camera — keyboard-only mode")
        return

    buf     = GestureBuffer(window=10, hold_frames=5)
    h_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame          = cv.flip(frame, 1)
        h_frame, _     = frame.shape[:2]
        rgb            = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img         = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result         = landmarker.detect(mp_img)

        raw_id = -1
        if result.hand_landmarks:
            lm_list = _calc_landmark_list(frame, result.hand_landmarks[0])
            raw_id  = int(classifier(_pre_process_landmark(lm_list)))

        stable_id = buf.update(raw_id)   # majority vote over last 10 frames

        action        = None
        overlay_text  = "no hand detected"
        overlay_color = (160, 160, 160)

        if stable_id != -1 and stable_id in _LABEL_MAP:
            action, overlay_text, overlay_color = _LABEL_MAP[stable_id]

        _set_gesture(action)

        # ── overlay ────────────────────────────────────────────────────────
        cv.putText(frame, overlay_text, (10, 48),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, overlay_color, 2, cv.LINE_AA)
        cv.putText(frame, f"raw={raw_id}  stable={stable_id}", (10, 85),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv.LINE_AA)
        cv.putText(frame,
                   "Open=Jump  Fist=Duck  Pointer=Run  ESC=close camera",
                   (8, h_frame - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1, cv.LINE_AA)
        cv.imshow("Gesture Controller", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    _set_gesture(None)
    cap.release()
    cv.destroyAllWindows()

# ── patch 1: menu screen — start game on any detected gesture ──────────────────
_original_draw_menu = DinoGame._draw_menu

def _patched_draw_menu(self):
    _original_draw_menu(self)
    self.root.after(150, self._gesture_menu_poll)

def _gesture_menu_poll(self):
    if self.state != 'menu':
        return
    if _get_gesture() is not None:
        self._start_game()
    else:
        self.root.after(150, self._gesture_menu_poll)

DinoGame._draw_menu         = _patched_draw_menu
DinoGame._gesture_menu_poll = _gesture_menu_poll

# ── patch 2: game loop — inject gesture keys each frame ───────────────────────
_original_loop = DinoGame._loop

def _patched_loop(self):
    action = _get_gesture()

    for k in getattr(self, '_injected', set()):
        self.keys.discard(k)
    self._injected = set()

    if action == 'jump':
        self.keys.add('Up')
        self._injected.add('Up')
    elif action == 'duck':
        self.keys.add('Down')
        self._injected.add('Down')

    _original_loop(self)

DinoGame._loop = _patched_loop

# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t = threading.Thread(target=_gesture_thread, daemon=True)
    t.start()
    DinoGame()
