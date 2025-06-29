import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Load keypoint data
df = pd.read_csv('model/keypoint_classifier/keypoint.csv', header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(42,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes (1–10)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save .h5 model
model.save("model/keypoint_classifier/keypoint_classifier.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model/keypoint_classifier/keypoint_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model trained and saved!")
