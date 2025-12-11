#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for trained MLP/LSTM models with webcam.
Uses MediaPipe Hands for hand detection and runs inference with trained models.
"""
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import argparse
from collections import deque
from pathlib import Path


def normalize_keypoints(landmarks):
    """
    Normalize keypoints relative to wrist position.
    Uses only X, Y coordinates (no Z) for stability.
    Matches preprocess.py exactly.
    
    Returns: 42 features (21 landmarks × 2 coordinates)
    """
    # Extract only X, Y coordinates (ignore Z)
    coords = [[lm.x, lm.y] for lm in landmarks.landmark]
    
    # Translate to wrist origin (landmark 0)
    base_x, base_y = coords[0]
    coords = [[c[0] - base_x, c[1] - base_y] for c in coords]
    
    # Flatten to 1D list
    flat = [val for coord in coords for val in coord]
    
    # Normalize by max absolute value
    max_val = max(abs(v) for v in flat)
    if max_val > 0:
        flat = [v / max_val for v in flat]
    
    return flat


def draw_landmarks(image, landmarks):
    """Draw hand landmarks on image."""
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )


def main():
    parser = argparse.ArgumentParser(description='Test ASL models with webcam')
    parser.add_argument('--model', type=str, choices=['mlp', 'lstm'], default='mlp',
                        help='Model to use (mlp or lstm)')
    parser.add_argument('--device', type=int, default=0,
                        help='Camera device number')
    parser.add_argument('--width', type=int, default=960,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=540,
                        help='Camera height')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug info about keypoints')
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent
    models_dir = project_root / 'models'
    
    # Load labels
    if args.model == 'lstm':
        labels_path = models_dir / 'lstm_labels.json'
    else:
        labels_path = models_dir / 'labels.json'
    
    print(f"Loading labels from: {labels_path}")
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    # Convert string keys to int
    labels = {int(k): v for k, v in labels.items()}
    print(f"Labels: {list(labels.values())}")
    
    # Load model
    if args.model == 'mlp':
        model_path = models_dir / 'mlp_model.keras'
    else:
        model_path = models_dir / 'lstm_model.keras'
    
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()
    
    # Get model input shape
    input_shape = model.input_shape
    print(f"\nModel input shape: {input_shape}")
    
    # For LSTM, we need sequence buffer
    SEQUENCE_LENGTH = 30 if args.model == 'lstm' else 1
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # Initialize MediaPipe Hands
    # Use settings closer to training (static_image_mode=True in training)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,  # False for video, but we use lower confidence
        max_num_hands=1,
        min_detection_confidence=0.5,  # Match training
        min_tracking_confidence=0.5
    )
    
    # Initialize camera
    print(f"\nStarting camera (device {args.device})...")
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\n" + "="*50)
    print(f"Testing {args.model.upper()} Model")
    print("="*50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current keypoints to file")
    print("  'd' - Toggle debug output")
    print("  'c' - Clear prediction history")
    
    # For smoothing predictions - use longer history for stability
    prediction_history = deque(maxlen=10)
    confidence_history = deque(maxlen=10)
    current_prediction = ""
    current_confidence = 0.0
    last_predictions = None  # Store raw predictions for display
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw info
        cv2.putText(frame, f"Model: {args.model.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            draw_landmarks(frame, hand_landmarks)
            
            # Extract and normalize keypoints
            keypoints = normalize_keypoints(hand_landmarks)
            
            # Debug output
            if args.debug:
                print(f"\nKeypoints (first 10 values): {keypoints[:10]}")
                print(f"Min: {min(keypoints):.4f}, Max: {max(keypoints):.4f}")
            
            if args.model == 'mlp':
                # MLP: Single frame inference
                input_data = np.array([keypoints], dtype=np.float32)
                predictions = model.predict(input_data, verbose=0)[0]
                last_predictions = predictions  # Store for display
                
                # Get top prediction
                pred_idx = np.argmax(predictions)
                confidence = predictions[pred_idx]
                pred_label = labels.get(pred_idx, f"Class {pred_idx}")
                
                # Smooth predictions using weighted voting
                prediction_history.append(pred_label)
                confidence_history.append(confidence)
                
                # Use most common prediction from history, weighted by confidence
                if len(prediction_history) >= 3:
                    # Count occurrences weighted by confidence
                    weighted_counts = {}
                    for p, c in zip(prediction_history, confidence_history):
                        if p not in weighted_counts:
                            weighted_counts[p] = 0
                        weighted_counts[p] += c
                    
                    # Get highest weighted prediction
                    best_pred = max(weighted_counts.keys(), key=lambda x: weighted_counts[x])
                    
                    # Calculate average confidence for this prediction
                    pred_confs = [c for p, c in zip(prediction_history, confidence_history) if p == best_pred]
                    avg_conf = np.mean(pred_confs) if pred_confs else 0
                    
                    current_prediction = best_pred
                    current_confidence = avg_conf
                
            else:
                # LSTM: Need sequence of frames
                sequence_buffer.append(keypoints)
                
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    input_data = np.array([list(sequence_buffer)], dtype=np.float32)
                    predictions = model.predict(input_data, verbose=0)[0]
                    last_predictions = predictions  # Store for display
                    
                    pred_idx = np.argmax(predictions)
                    confidence = predictions[pred_idx]
                    pred_label = labels.get(pred_idx, f"Class {pred_idx}")
                    
                    current_prediction = pred_label
                    current_confidence = confidence
                else:
                    cv2.putText(frame, f"Buffering: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display prediction
            if current_prediction:
                color = (0, 255, 0) if current_confidence > 0.7 else (0, 255, 255)
                cv2.putText(frame, f"Prediction: {current_prediction}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(frame, f"Confidence: {current_confidence:.2%}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw large letter
                cv2.putText(frame, current_prediction, (frame.shape[1] - 150, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 5)
                
                # Show top 5 predictions
                if last_predictions is not None:
                    top5_idx = np.argsort(last_predictions)[-5:][::-1]
                    y_pos = 150
                    for idx in top5_idx:
                        label = labels.get(idx, f"Class {idx}")
                        conf = last_predictions[idx]
                        bar_width = int(conf * 200)
                        
                        # Color based on confidence
                        if conf > 0.5:
                            bar_color = (0, 200, 0)  # Green
                        elif conf > 0.2:
                            bar_color = (0, 200, 200)  # Yellow
                        else:
                            bar_color = (100, 100, 100)  # Gray
                        
                        cv2.rectangle(frame, (10, y_pos - 15), (10 + bar_width, y_pos + 5), 
                                      bar_color, -1)
                        cv2.putText(frame, f"{label}: {conf:.1%}", (15, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_pos += 25
        else:
            cv2.putText(frame, "No hand detected", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            sequence_buffer.clear()
            prediction_history.clear()
        
        # Show frame
        cv2.imshow('ASL Model Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            args.debug = not args.debug
            print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
        elif key == ord('c'):
            prediction_history.clear()
            confidence_history.clear()
            current_prediction = ""
            print("Cleared prediction history")
        elif key == ord('s') and results.multi_hand_landmarks:
            # Save current keypoints
            keypoints = normalize_keypoints(results.multi_hand_landmarks[0])
            save_path = project_root / 'debug_keypoints.json'
            with open(save_path, 'w') as f:
                json.dump({
                    'keypoints': keypoints,
                    'prediction': current_prediction,
                    'confidence': float(current_confidence)
                }, f, indent=2)
            print(f"Saved keypoints to {save_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nDone!")


if __name__ == '__main__':
    main()

