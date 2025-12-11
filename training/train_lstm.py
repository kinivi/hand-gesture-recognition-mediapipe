"""
LSTM classifier for ASL alphabet recognition.
Trains on sequences of MediaPipe keypoints for dynamic gesture recognition.
Handles both collected sequences and augmented static data.
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


SEQUENCE_LENGTH = 30  # Must match collect_sequences.py


def load_sequence_data(csv_path):
    """
    Load sequence data from CSV.
    Returns X with shape (num_sequences, sequence_length, 63) and labels.
    """
    df = pd.read_csv(csv_path)
    
    # Get unique sequence IDs
    sequence_ids = df['sequence_id'].unique()
    
    sequences = []
    labels = []
    
    for seq_id in sequence_ids:
        seq_data = df[df['sequence_id'] == seq_id].sort_values('frame_idx')
        
        # Extract keypoint features (columns x0 through z20)
        feature_cols = [col for col in df.columns if col not in ['sequence_id', 'frame_idx', 'label']]
        keypoints = seq_data[feature_cols].values
        
        # Ensure correct sequence length
        if len(keypoints) == SEQUENCE_LENGTH:
            sequences.append(keypoints)
            labels.append(seq_data['label'].iloc[0])
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels)
    
    return X, y


def augment_static_keypoints(static_csv_path, num_sequences_per_class=100):
    """
    Create synthetic sequences from static keypoint data.
    Adds small temporal variations to simulate movement.
    
    This allows LSTM training even without recorded sequences.
    """
    df = pd.read_csv(static_csv_path)
    
    feature_cols = [col for col in df.columns if col != 'label']
    
    sequences = []
    labels = []
    
    for label in df['label'].unique():
        class_data = df[df['label'] == label][feature_cols].values
        
        for _ in range(num_sequences_per_class):
            # Pick a random sample as base
            base_idx = np.random.randint(len(class_data))
            base_keypoints = class_data[base_idx]
            
            # Create sequence with small random variations
            sequence = []
            for frame in range(SEQUENCE_LENGTH):
                # Add small temporal noise to simulate natural movement
                noise = np.random.normal(0, 0.02, size=base_keypoints.shape)
                frame_keypoints = base_keypoints + noise
                sequence.append(frame_keypoints)
            
            sequences.append(sequence)
            labels.append(label)
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels)
    
    return X, y


def create_lstm_model(input_shape, num_classes):
    """
    Create LSTM model for sequence classification.
    Input shape: (sequence_length, num_features) - e.g., (30, 42) for 2D keypoints
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Bidirectional LSTM layers
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_model(X_train, y_train, X_val, y_val, label_encoder, output_dir):
    """Train the LSTM model with early stopping."""
    
    num_classes = len(label_encoder.classes_)
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, 63)
    
    print(f"\nModel Configuration:")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training sequences: {len(X_train)}")
    
    # Create model
    model = create_lstm_model(input_shape, num_classes)
    
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
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'lstm_best.keras'),
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
        batch_size=32,
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
    axes[0].set_title('LSTM Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('LSTM Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lstm_training_history.png', dpi=150)
    plt.close()
    print(f"Saved training plot to {output_dir / 'lstm_training_history.png'}")


def save_label_encoder(label_encoder, output_dir):
    """Save label encoder classes for inference."""
    labels = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(output_dir / 'lstm_labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"Saved label mapping to {output_dir / 'lstm_labels.json'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM classifier for ASL')
    parser.add_argument('--sequences', type=str, default='data/sequences.csv',
                        help='Path to sequences CSV file (from collect_sequences.py)')
    parser.add_argument('--static', type=str, default='data/keypoints.csv',
                        help='Path to static keypoints CSV (for augmentation)')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for model')
    parser.add_argument('--augment-only', action='store_true',
                        help='Use only augmented static data (no real sequences)')
    parser.add_argument('--augment-count', type=int, default=100,
                        help='Number of augmented sequences per class')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    sequences_path = project_root / args.sequences
    static_path = project_root / args.static
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("ASL LSTM Classifier Training")
    print("="*50)
    
    # Load or generate data
    if args.augment_only or not sequences_path.exists():
        print(f"\nGenerating augmented sequences from static keypoints...")
        if not static_path.exists():
            print(f"Error: Static keypoints file not found: {static_path}")
            print("Run preprocess.py first to extract keypoints from images.")
            return
        X, y = augment_static_keypoints(static_path, args.augment_count)
        print(f"Generated {len(X)} augmented sequences")
    else:
        print(f"\nLoading recorded sequences from {sequences_path}...")
        X, y = load_sequence_data(sequences_path)
        print(f"Loaded {len(X)} sequences")
        
        # Optionally combine with augmented data
        if static_path.exists():
            print("Augmenting with static keypoints...")
            X_aug, y_aug = augment_static_keypoints(static_path, args.augment_count // 2)
            X = np.concatenate([X, X_aug])
            y = np.concatenate([y, y_aug])
            print(f"Total sequences after augmentation: {len(X)}")
    
    print(f"Sequence shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
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
        print(f"Remaining sequences: {len(X)}")
    
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
    print(f"  Training: {len(X_train)} sequences")
    print(f"  Validation: {len(X_val)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # Train model
    print("\nTraining LSTM model...")
    model, history = train_model(X_train, y_train, X_val, y_val, label_encoder, output_dir)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save artifacts
    model.save(output_dir / 'lstm_model.keras')
    print(f"\nSaved model to {output_dir / 'lstm_model.keras'}")
    
    # Also save in H5 format for TensorFlow.js conversion
    model.save(output_dir / 'lstm_model.h5')
    print(f"Saved H5 model to {output_dir / 'lstm_model.h5'}")
    
    save_label_encoder(label_encoder, output_dir)
    plot_training_history(history, output_dir)
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == '__main__':
    main()

