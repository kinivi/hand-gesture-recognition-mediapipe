"""
SVM (Support Vector Machine) classifier for ASL alphabet recognition.
Trains on extracted MediaPipe keypoints with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns


def load_keypoints(csv_path):
    """Load keypoints from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values
    
    return X, y


def train_svm_with_gridsearch(X_train, y_train, quick_mode=False):
    """
    Train SVM with grid search for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        quick_mode: If True, use smaller parameter grid for faster training
    """
    
    if quick_mode:
        # Smaller grid for quick testing
        param_grid = {
            'C': [1, 10],
            'gamma': ['scale', 0.01],
            'kernel': ['rbf']
        }
        cv = 3
    else:
        # Full grid search
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly']
        }
        cv = 5
    
    print("\nStarting Grid Search...")
    print(f"Parameter grid: {param_grid}")
    print(f"Cross-validation folds: {cv}")
    
    svm = SVC(random_state=42, probability=True)
    
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def train_svm_simple(X_train, y_train):
    """Train SVM with default good parameters for faster training."""
    print("\nTraining SVM with optimized parameters...")
    
    svm = SVC(
        C=10,
        gamma='scale',
        kernel='rbf',
        random_state=42,
        probability=True,
        verbose=True
    )
    
    svm.fit(X_train, y_train)
    return svm, {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}


def evaluate_model(model, X_test, y_test, label_encoder, output_dir):
    """Evaluate SVM model on test set."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print(report)
    
    # Save classification report
    with open(output_dir / 'svm_classification_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_, output_dir)
    
    return accuracy


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('SVM Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'svm_confusion_matrix.png', dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {output_dir / 'svm_confusion_matrix.png'}")


def save_model(model, scaler, label_encoder, best_params, output_dir):
    """Save SVM model and associated artifacts."""
    # Save model
    joblib.dump(model, output_dir / 'svm_model.pkl')
    print(f"Saved model to {output_dir / 'svm_model.pkl'}")
    
    # Save scaler
    joblib.dump(scaler, output_dir / 'svm_scaler.pkl')
    print(f"Saved scaler to {output_dir / 'svm_scaler.pkl'}")
    
    # Save label encoder
    joblib.dump(label_encoder, output_dir / 'svm_label_encoder.pkl')
    print(f"Saved label encoder to {output_dir / 'svm_label_encoder.pkl'}")
    
    # Save metadata
    metadata = {
        'best_params': best_params,
        'classes': list(label_encoder.classes_),
        'n_features': model.n_features_in_,
        'n_classes': len(label_encoder.classes_)
    }
    with open(output_dir / 'svm_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {output_dir / 'svm_metadata.json'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SVM classifier for ASL')
    parser.add_argument('--data', type=str, default='data/keypoints.csv',
                        help='Path to keypoints CSV file')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for model')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--grid-search', action='store_true',
                        help='Use grid search for hyperparameter tuning')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with smaller parameter grid')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    data_path = project_root / args.data
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("ASL SVM Classifier Training")
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=args.test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if args.grid_search:
        model, best_params = train_svm_with_gridsearch(
            X_train_scaled, y_train, quick_mode=args.quick
        )
    else:
        model, best_params = train_svm_simple(X_train_scaled, y_train)
    
    # Evaluate
    evaluate_model(model, X_test_scaled, y_test, label_encoder, output_dir)
    
    # Save
    save_model(model, scaler, label_encoder, best_params, output_dir)
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == '__main__':
    main()

