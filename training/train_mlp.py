"""
MLP (Multi-Layer Perceptron) classifier for ASL alphabet recognition.
Trains on extracted MediaPipe keypoints.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json


def load_keypoints(csv_path):
    """Load keypoints from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values
    
    return X, y


def create_mlp_model(input_dim, num_classes):
    """
    Create MLP model architecture.
    Input: 63 keypoint features
    Output: num_classes (29 for full ASL alphabet + del, nothing, space)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_model(X_train, y_train, X_val, y_val, label_encoder, output_dir):
    """Train the MLP model with early stopping."""
    
    num_classes = len(label_encoder.classes_)
    input_dim = X_train.shape[1]
    
    print(f"\nModel Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {list(label_encoder.classes_)}")
    
    # Create model
    model = create_mlp_model(input_dim, num_classes)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'mlp_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model on test set."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(label_encoder.classes_):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_test[mask])
            print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    return accuracy


def plot_training_history(history, output_dir):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mlp_training_history.png', dpi=150)
    plt.close()
    print(f"Saved training plot to {output_dir / 'mlp_training_history.png'}")


def save_label_encoder(label_encoder, output_dir):
    """Save label encoder classes for inference."""
    labels = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(output_dir / 'labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"Saved label mapping to {output_dir / 'labels.json'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MLP classifier for ASL')
    parser.add_argument('--data', type=str, default='data/keypoints.csv',
                        help='Path to keypoints CSV file')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for model')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    data_path = project_root / args.data
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("ASL MLP Classifier Training")
    print("="*50)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    X, y = load_keypoints(data_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Check class distribution and filter out classes with too few samples
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("\nClass distribution:")
    min_samples_required = 10  # Minimum samples needed per class
    classes_to_remove = []
    for cls, count in sorted(class_counts.items()):
        status = "" if count >= min_samples_required else " (REMOVING - too few samples)"
        print(f"  {cls}: {count}{status}")
        if count < min_samples_required:
            classes_to_remove.append(cls)
    
    # Filter out classes with too few samples
    if classes_to_remove:
        print(f"\nRemoving {len(classes_to_remove)} classes with < {min_samples_required} samples")
        mask = ~np.isin(y, classes_to_remove)
        X = X[mask]
        y = y[mask]
        print(f"Remaining samples: {len(X)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=args.test_size * 2, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train model
    print("\nTraining MLP model...")
    model, history = train_model(X_train, y_train, X_val, y_val, label_encoder, output_dir)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save artifacts
    model.save(output_dir / 'mlp_model.keras')
    print(f"\nSaved model to {output_dir / 'mlp_model.keras'}")
    
    # Also save in H5 format for TensorFlow.js conversion
    model.save(output_dir / 'mlp_model.h5')
    print(f"Saved H5 model to {output_dir / 'mlp_model.h5'}")
    
    save_label_encoder(label_encoder, output_dir)
    plot_training_history(history, output_dir)
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == '__main__':
    main()

