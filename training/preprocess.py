"""
Keypoint extraction from ASL Alphabet dataset using MediaPipe Hands.
Extracts 21 hand landmarks (x, y) = 42 features per image.
Uses 2D only (no Z) for better stability between static images and live video.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def normalize_keypoints(landmarks):
    """
    Normalize keypoints relative to wrist position.
    Uses only X, Y coordinates (no Z) for stability.
    Normalizes by max absolute value (matches reference implementation).
    
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


def extract_keypoints_from_image(image_path, hands):
    """
    Extract hand keypoints from a single image.
    Returns None if no hand is detected.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Take the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        return normalize_keypoints(hand_landmarks)
    
    return None


def process_dataset(data_dir, output_path, max_per_class=None):
    """
    Process entire ASL alphabet dataset and extract keypoints.
    
    Args:
        data_dir: Path to asl_alphabet_train/asl_alphabet_train/
        output_path: Path to save the CSV file
        max_per_class: Maximum images to process per class (for testing)
    """
    data_dir = Path(data_dir)
    
    # Get all class directories
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes: {classes}")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    all_keypoints = []
    all_labels = []
    skipped = 0
    
    for class_name in classes:
        class_dir = data_dir / class_name
        image_files = list(class_dir.glob("*.jpg"))
        
        if max_per_class:
            image_files = image_files[:max_per_class]
        
        print(f"\nProcessing class '{class_name}' ({len(image_files)} images)...")
        
        for img_path in tqdm(image_files, desc=class_name):
            keypoints = extract_keypoints_from_image(img_path, hands)
            
            if keypoints is not None:
                all_keypoints.append(keypoints)
                all_labels.append(class_name)
            else:
                skipped += 1
    
    hands.close()
    
    # Create DataFrame
    # Column names: x0, y0, x1, y1, ..., x20, y20, label (42 features + label)
    columns = []
    for i in range(21):
        columns.extend([f'x{i}', f'y{i}'])
    columns.append('label')
    
    # Combine keypoints and labels
    data = np.column_stack([np.array(all_keypoints), np.array(all_labels)])
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total samples: {len(all_keypoints)}")
    print(f"Features per sample: 42 (21 landmarks × 2 coords)")
    print(f"Skipped (no hand detected): {skipped}")
    print(f"Saved to: {output_path}")
    print(f"{'='*50}")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract keypoints from ASL dataset')
    parser.add_argument('--data-dir', type=str, 
                        default='archive/asl_alphabet_train/asl_alphabet_train',
                        help='Path to training data directory')
    parser.add_argument('--output', type=str, 
                        default='data/keypoints.csv',
                        help='Output CSV file path')
    parser.add_argument('--max-per-class', type=int, default=None,
                        help='Max images per class (for testing)')
    
    args = parser.parse_args()
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    data_dir = project_root / args.data_dir
    output_path = project_root / args.output
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    process_dataset(data_dir, output_path, args.max_per_class)


if __name__ == '__main__':
    main()
