import tensorflow as tf

NUM_CLASSES = 5
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

# Wrap the Sequential model in a Functional Model
seq_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

inputs = tf.keras.Input(shape=(42,))
outputs = seq_model(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Save the keras model normally
model.save('model/keypoint_classifier/keypoint_classifier.keras')

# TFLite conversion (quantized)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = False  # safer on Apple Silicon

tflite_quantized_model = converter.convert()

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)

print("Saved TFLite model:", tflite_save_path)
