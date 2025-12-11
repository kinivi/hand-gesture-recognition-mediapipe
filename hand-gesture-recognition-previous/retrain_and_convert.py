#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retrain the keypoint classifier with all classes in keypoint.csv
and convert to TensorFlow.js format.
"""
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def main():
    # Paths
    dataset_path = 'model/keypoint_classifier/keypoint.csv'
    labels_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    output_dir = 'tfjs_model_retrained'
    
    print("=" * 60)
    print("Retraining Keypoint Classifier")
    print("=" * 60)
    
    # Load training data
    print(f"\nLoading data from: {dataset_path}")
    X_dataset = np.loadtxt(dataset_path, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset_path, delimiter=',', dtype='int32', usecols=(0))
    
    # Determine number of classes from data
    NUM_CLASSES = len(np.unique(y_dataset))
    print(f"Found {len(X_dataset)} samples")
    print(f"Found {NUM_CLASSES} unique classes: {sorted(np.unique(y_dataset))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build model (same architecture as original notebook)
    print("\nBuilding model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.summary()
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\nTraining model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as Keras model first
    keras_path = os.path.join(output_dir, 'keypoint_classifier.keras')
    model.save(keras_path)
    print(f"\nKeras model saved to: {keras_path}")
    
    # Convert to TensorFlow.js
    print("\nConverting to TensorFlow.js format...")
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, output_dir)
    print(f"TensorFlow.js model saved to: {output_dir}/")
    
    # Read and save labels
    print(f"\nReading labels from: {labels_path}")
    with open(labels_path, 'r', encoding='utf-8-sig') as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Only use labels for the classes we have
    labels = labels[:NUM_CLASSES]
    labels_json = {str(i): label for i, label in enumerate(labels)}
    
    labels_output = os.path.join(output_dir, 'labels.json')
    with open(labels_output, 'w') as f:
        json.dump(labels_json, f, indent=2)
    
    print(f"\nLabels ({len(labels)} classes):")
    for i, label in enumerate(labels):
        print(f"  {i}: {label}")
    
    print(f"\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"\nFiles in {output_dir}/:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")
    
    print(f"\nTo use in frontend, copy to frontend/public/models/keypoint/:")
    print(f"  cp {output_dir}/model.json frontend/public/models/keypoint/")
    print(f"  cp {output_dir}/group*.bin frontend/public/models/keypoint/")
    print(f"  cp {output_dir}/labels.json frontend/public/models/keypoint/")

if __name__ == '__main__':
    main()

