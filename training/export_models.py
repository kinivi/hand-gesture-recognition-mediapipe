"""
Export trained models to TensorFlow.js format for browser deployment.
Converts MLP and LSTM models from Keras to TFJS format.
"""

import subprocess
import sys
from pathlib import Path
import json
import shutil


def convert_keras_to_tfjs(model_path, output_dir, model_name):
    """
    Convert Keras model to TensorFlow.js format.
    
    Args:
        model_path: Path to .h5 or .keras model file
        output_dir: Directory to save TFJS model
        model_name: Name for the output folder
    """
    output_path = output_dir / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConverting {model_path} to TensorFlow.js...")
    
    try:
        # Use tensorflowjs_converter CLI
        cmd = [
            sys.executable, '-m', 'tensorflowjs.converters.keras_h5_to_tfjs',
            str(model_path),
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  Success! Saved to {output_path}")
            return True
        else:
            # Try alternative conversion method
            print("  Trying alternative conversion method...")
            cmd = [
                'tensorflowjs_converter',
                '--input_format=keras',
                str(model_path),
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  Success! Saved to {output_path}")
                return True
            else:
                print(f"  Error: {result.stderr}")
                return False
                
    except Exception as e:
        print(f"  Error during conversion: {e}")
        return False


def copy_label_files(models_dir, tfjs_dir):
    """Copy label JSON files to TFJS output directory."""
    label_files = list(models_dir.glob('*labels*.json'))
    
    for label_file in label_files:
        dest = tfjs_dir / label_file.name
        shutil.copy(label_file, dest)
        print(f"Copied {label_file.name} to {tfjs_dir}")


def create_model_manifest(tfjs_dir, models_exported):
    """Create a manifest file listing available models."""
    manifest = {
        'models': models_exported,
        'version': '1.0.0',
        'description': 'ASL Alphabet Classifier Models'
    }
    
    with open(tfjs_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nCreated manifest at {tfjs_dir / 'manifest.json'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Export models to TensorFlow.js')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--output', type=str, default='models/tfjs',
                        help='Output directory for TFJS models')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    models_dir = project_root / args.models_dir
    tfjs_dir = project_root / args.output
    tfjs_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("Exporting Models to TensorFlow.js")
    print("="*50)
    
    models_exported = []
    
    # Export MLP model
    mlp_h5 = models_dir / 'mlp_model.h5'
    if mlp_h5.exists():
        if convert_keras_to_tfjs(mlp_h5, tfjs_dir, 'mlp'):
            models_exported.append({
                'name': 'mlp',
                'type': 'MLP',
                'path': 'mlp/model.json',
                'labels': 'labels.json'
            })
    else:
        print(f"\nMLP model not found at {mlp_h5}")
    
    # Export LSTM model
    lstm_h5 = models_dir / 'lstm_model.h5'
    if lstm_h5.exists():
        if convert_keras_to_tfjs(lstm_h5, tfjs_dir, 'lstm'):
            models_exported.append({
                'name': 'lstm',
                'type': 'LSTM',
                'path': 'lstm/model.json',
                'labels': 'lstm_labels.json',
                'sequence_length': 30
            })
    else:
        print(f"\nLSTM model not found at {lstm_h5}")
    
    # Copy label files
    print("\n" + "-"*50)
    copy_label_files(models_dir, tfjs_dir)
    
    # Create manifest
    create_model_manifest(tfjs_dir, models_exported)
    
    # Summary
    print("\n" + "="*50)
    print("Export Summary")
    print("="*50)
    print(f"Models exported: {len(models_exported)}")
    for model in models_exported:
        print(f"  - {model['name']} ({model['type']})")
    print(f"\nOutput directory: {tfjs_dir}")
    
    # Note about SVM
    svm_pkl = models_dir / 'svm_model.pkl'
    if svm_pkl.exists():
        print("\n" + "-"*50)
        print("Note: SVM model (svm_model.pkl) cannot be directly converted to TensorFlow.js.")
        print("Options for SVM deployment:")
        print("  1. Use a Python backend (Flask/FastAPI) to serve SVM predictions")
        print("  2. Convert to ONNX format and use onnxruntime-web")
        print("  3. Use sklearn-porter to generate JavaScript code (limited support)")
    
    print("\n" + "="*50)
    print("Export complete!")
    print("="*50)


if __name__ == '__main__':
    main()

