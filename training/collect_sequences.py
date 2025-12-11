"""
Webcam-based sequence collector for LSTM training.
Records sequences of hand keypoints for dynamic ASL letters (J, Z, etc.).

Controls:
- Press a letter key (a-z) to start recording for that letter
- Recording automatically stops after SEQUENCE_LENGTH frames
- Press 'q' to quit
- Press 's' to save collected data
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json


# Configuration
SEQUENCE_LENGTH = 30  # Number of frames per sequence
FPS_TARGET = 30


def normalize_keypoints(landmarks):
    """
    Normalize keypoints relative to wrist position.
    Uses only X, Y coordinates (no Z) for stability.
    Normalizes by max absolute value.
    Returns 42 features (21 landmarks × 2 coordinates).
    """
    # Extract only X, Y (no Z)
    coords = [[lm.x, lm.y] for lm in landmarks.landmark]
    
    # Translate to wrist origin
    base_x, base_y = coords[0]
    coords = [[c[0] - base_x, c[1] - base_y] for c in coords]
    
    # Flatten
    flat = [val for coord in coords for val in coord]
    
    # Normalize by max absolute value
    max_val = max(abs(v) for v in flat)
    if max_val > 0:
        flat = [v / max_val for v in flat]
    
    return np.array(flat)


def draw_hand_landmarks(image, hand_landmarks, mp_drawing, mp_hands):
    """Draw hand landmarks on image."""
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )


def create_info_overlay(image, recording, current_letter, frame_count, sequences_count):
    """Add information overlay to the image."""
    h, w = image.shape[:2]
    
    # Status bar background
    cv2.rectangle(image, (0, 0), (w, 80), (0, 0, 0), -1)
    
    # Recording status
    if recording:
        status_color = (0, 0, 255)  # Red
        status_text = f"RECORDING: {current_letter.upper()} ({frame_count}/{SEQUENCE_LENGTH})"
        # Blinking recording indicator
        if frame_count % 10 < 5:
            cv2.circle(image, (30, 30), 15, (0, 0, 255), -1)
    else:
        status_color = (0, 255, 0)  # Green
        status_text = "Ready - Press a letter key to record"
    
    cv2.putText(image, status_text, (60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Sequence count
    cv2.putText(image, f"Sequences collected: {sequences_count}", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Instructions at bottom
    cv2.putText(image, "Press letter to record | 's' to save | 'q' to quit", 
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return image


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect gesture sequences for LSTM')
    parser.add_argument('--output', type=str, default='data/sequences.csv',
                        help='Output CSV file path')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("ASL Sequence Collector for LSTM Training")
    print("="*50)
    print(f"\nSequence length: {SEQUENCE_LENGTH} frames")
    print(f"Output: {output_path}")
    print("\nControls:")
    print("  - Press a letter key (a-z) to start recording")
    print("  - Press 's' to save collected data")
    print("  - Press 'q' to quit")
    print("\n")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Data storage
    all_sequences = []  # List of (sequence, label) tuples
    current_sequence = []
    recording = False
    current_letter = None
    
    print("Camera ready. Window opened.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw landmarks if hand detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks, mp_drawing, mp_hands)
                
                # If recording, capture keypoints
                if recording:
                    keypoints = normalize_keypoints(hand_landmarks)
                    current_sequence.append(keypoints)
                    
                    # Check if sequence complete
                    if len(current_sequence) >= SEQUENCE_LENGTH:
                        # Save sequence
                        all_sequences.append((np.array(current_sequence), current_letter))
                        print(f"  Recorded sequence for '{current_letter.upper()}' "
                              f"(Total: {len(all_sequences)})")
                        
                        # Reset
                        current_sequence = []
                        recording = False
                        current_letter = None
        else:
            # No hand detected during recording - warn but continue
            if recording:
                cv2.putText(frame, "No hand detected!", (200, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Create overlay
        frame = create_info_overlay(
            frame, 
            recording, 
            current_letter or '', 
            len(current_sequence),
            len(all_sequences)
        )
        
        # Show frame
        cv2.imshow('ASL Sequence Collector', frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save data
            if len(all_sequences) > 0:
                save_sequences(all_sequences, output_path)
            else:
                print("No sequences to save!")
        elif ord('a') <= key <= ord('z') and not recording:
            # Start recording for this letter
            current_letter = chr(key)
            current_sequence = []
            recording = True
            print(f"\nRecording for letter '{current_letter.upper()}'...")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Final save prompt
    if len(all_sequences) > 0:
        print(f"\n{len(all_sequences)} sequences collected.")
        save = input("Save before exiting? (y/n): ").lower().strip()
        if save == 'y':
            save_sequences(all_sequences, output_path)


def save_sequences(sequences, output_path):
    """Save collected sequences to CSV."""
    
    # Each row: sequence_id, frame_idx, 42 keypoint features (x,y only), label
    rows = []
    
    for seq_idx, (sequence, label) in enumerate(sequences):
        for frame_idx, keypoints in enumerate(sequence):
            row = [seq_idx, frame_idx] + list(keypoints) + [label]
            rows.append(row)
    
    # Create column names (42 features: x0,y0,x1,y1,...,x20,y20)
    columns = ['sequence_id', 'frame_idx']
    for i in range(21):
        columns.extend([f'x{i}', f'y{i}'])
    columns.append('label')
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(sequences)} sequences ({len(rows)} frames) to {output_path}")
    
    # Summary
    label_counts = df.groupby('label')['sequence_id'].nunique()
    print("\nSequences per letter:")
    for label, count in label_counts.items():
        print(f"  {label.upper()}: {count}")


if __name__ == '__main__':
    main()

