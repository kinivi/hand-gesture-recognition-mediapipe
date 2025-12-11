#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export models to TensorFlow.js format with compatibility fix for Keras 3.x
"""
import os
import json
import shutil
from pathlib import Path
import tensorflow as tf
import tensorflowjs as tfjs


def fix_model_json(model_json_path):
    """Fix the model.json to be compatible with TensorFlow.js"""
    with open(model_json_path, 'r') as f:
        model_data = json.load(f)
    
    # Fix the InputLayer config
    if 'modelTopology' in model_data:
        topology = model_data['modelTopology']
        if 'model_config' in topology:
            config = topology['model_config']
            if 'config' in config and 'layers' in config['config']:
                for layer in config['config']['layers']:
                    if layer.get('class_name') == 'InputLayer':
                        layer_config = layer.get('config', {})
                        # Convert batch_shape to batchInputShape
                        if 'batch_shape' in layer_config:
                            layer_config['batchInputShape'] = layer_config.pop('batch_shape')
                        # Also handle batch_input_shape
                        if 'batch_input_shape' in layer_config:
                            layer_config['batchInputShape'] = layer_config.pop('batch_input_shape')
    
    with open(model_json_path, 'w') as f:
        json.dump(model_data, f)
    
    print(f"  Fixed {model_json_path}")


def export_model(model_path, output_dir, model_name):
    """Export a single model to TensorFlow.js format"""
    print(f"\nExporting {model_name}...")
    
    # Load model
    print(f"  Loading from {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()
    
    # Create output directory
    output_path = output_dir / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export to TensorFlow.js
    print(f"  Converting to TensorFlow.js format...")
    tfjs.converters.save_keras_model(model, str(output_path))
    
    # Fix the model.json for TensorFlow.js compatibility
    model_json_path = output_path / 'model.json'
    fix_model_json(model_json_path)
    
    print(f"  Saved to {output_path}")
    return output_path


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / 'models'
    frontend_models_dir = project_root / 'frontend' / 'public' / 'models'
    
    print("=" * 60)
    print("Exporting Models to TensorFlow.js (with Keras 3.x fix)")
    print("=" * 60)
    
    # Export MLP
    mlp_path = models_dir / 'mlp_model.keras'
    if mlp_path.exists():
        mlp_output = export_model(mlp_path, frontend_models_dir, 'mlp')
        
        # Copy labels
        labels_src = models_dir / 'labels.json'
        if labels_src.exists():
            shutil.copy(labels_src, frontend_models_dir / 'labels.json')
            print(f"  Copied labels.json")
    else:
        print(f"MLP model not found at {mlp_path}")
    
    # Export LSTM
    lstm_path = models_dir / 'lstm_model.keras'
    if lstm_path.exists():
        lstm_output = export_model(lstm_path, frontend_models_dir, 'lstm')
        
        # Copy LSTM labels
        lstm_labels_src = models_dir / 'lstm_labels.json'
        if lstm_labels_src.exists():
            shutil.copy(lstm_labels_src, frontend_models_dir / 'lstm_labels.json')
            print(f"  Copied lstm_labels.json")
    else:
        print(f"LSTM model not found at {lstm_path}")
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nModels exported to: {frontend_models_dir}")


if __name__ == '__main__':
    main()

