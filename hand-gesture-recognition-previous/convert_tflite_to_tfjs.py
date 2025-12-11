#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert keypoint_classifier TFLite model to TensorFlow.js format

The TFLite model is the correct one used by app.py.
The Keras model has only 10 classes but the TFLite has 25.
"""
import os
import json
import subprocess
import sys

def main():
    tflite_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
    labels_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    output_dir = 'tfjs_model_from_tflite'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting TFLite model: {tflite_path}")
    print(f"Output directory: {output_dir}")
    
    # Convert using tensorflowjs_converter CLI
    # This is the correct way to convert TFLite to TFJS
    cmd = [
        sys.executable, '-m', 'tensorflowjs.converters.converter',
        '--input_format=tf_lite',
        '--output_format=tfjs_graph_model',
        tflite_path,
        output_dir
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        print("\nTrying alternative method with tfjs converter...")
        
        # Alternative: use tensorflowjs_converter directly
        cmd2 = [
            'tensorflowjs_converter',
            '--input_format=tf_lite',
            '--output_format=tfjs_graph_model', 
            tflite_path,
            output_dir
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            print(f"Error: {result2.stderr}")
            return
    
    print(f"\n✅ Model converted successfully!")
    
    # Read and save labels
    print(f"\nReading labels from: {labels_path}")
    with open(labels_path, 'r', encoding='utf-8-sig') as f:
        labels = [line.strip() for line in f.readlines()]
    
    labels_json = {str(i): label for i, label in enumerate(labels)}
    
    labels_output = os.path.join(output_dir, 'labels.json')
    with open(labels_output, 'w') as f:
        json.dump(labels_json, f, indent=2)
    
    print(f"\nLabels ({len(labels)} classes):")
    for i, label in enumerate(labels):
        print(f"  {i}: {label}")
    
    print(f"\n✅ Labels saved to: {labels_output}")
    print(f"\nModel files in {output_dir}/:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")

if __name__ == '__main__':
    main()

