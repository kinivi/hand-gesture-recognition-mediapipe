#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare captured keypoints with training data to diagnose issues.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


def main():
    project_root = Path(__file__).parent
    
    # Load training data
    print("Loading training data...")
    df = pd.read_csv(project_root / 'data/keypoints.csv')
    X_train = df.drop('label', axis=1).values.astype(np.float32)
    y_train = df['label'].values
    
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Try to load debug keypoints
    debug_path = project_root / 'debug_keypoints.json'
    if not debug_path.exists():
        print(f"\nNo debug keypoints found at {debug_path}")
        print("Run test_models.py and press 's' to save keypoints")
        return
    
    with open(debug_path) as f:
        debug_data = json.load(f)
    
    keypoints = np.array(debug_data['keypoints'], dtype=np.float32)
    saved_pred = debug_data.get('prediction', 'Unknown')
    saved_conf = debug_data.get('confidence', 0)
    
    print(f"\nLoaded debug keypoints:")
    print(f"  Saved prediction: {saved_pred} ({saved_conf:.2%})")
    print(f"  Keypoints shape: {keypoints.shape}")
    print(f"  First 10 values: {keypoints[:10]}")
    print(f"  Min: {keypoints.min():.4f}, Max: {keypoints.max():.4f}")
    
    # Compare with training data statistics
    print("\n--- Comparison with Training Data ---")
    
    # Find nearest neighbors in training data
    print("\nFinding nearest neighbors in training data...")
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X_train)
    
    distances, indices = nn.kneighbors([keypoints])
    
    print("\nTop 5 nearest training samples:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        label = y_train[idx]
        print(f"  {i+1}. Label: {label}, Distance: {dist:.4f}")
    
    # Check per-class average distances
    print("\n--- Average Distance to Each Class ---")
    class_distances = {}
    for label in np.unique(y_train):
        mask = y_train == label
        class_samples = X_train[mask]
        dists = np.linalg.norm(class_samples - keypoints, axis=1)
        class_distances[label] = {
            'mean': np.mean(dists),
            'min': np.min(dists),
            'std': np.std(dists)
        }
    
    # Sort by mean distance
    sorted_classes = sorted(class_distances.items(), key=lambda x: x[1]['mean'])
    
    print("\nTop 10 closest classes by mean distance:")
    for label, stats in sorted_classes[:10]:
        print(f"  {label}: mean={stats['mean']:.4f}, min={stats['min']:.4f}, std={stats['std']:.4f}")
    
    # Compare specific coordinate patterns
    print("\n--- Coordinate Analysis ---")
    
    # Check if thumb is extended (compare x4 to x0)
    thumb_extended = keypoints[8] > 0.3  # x4 (thumb tip)
    print(f"Thumb extended (x4 > 0.3): {thumb_extended} (x4={keypoints[8]:.4f})")
    
    # Check if fingers are curled (y values of fingertips should be closer to wrist)
    fingertip_y = [keypoints[i*2+1] for i in [4, 8, 12, 16, 20]]  # y4, y8, y12, y16, y20
    print(f"Fingertip Y values: {[f'{y:.3f}' for y in fingertip_y]}")


if __name__ == '__main__':
    main()

