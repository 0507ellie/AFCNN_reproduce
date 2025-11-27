"""
Super Simple Image Testing Script
Change the image_path variable to test your own images!
"""

import sys
import numpy as np
import cv2
from tensorflow import keras
from model_official import ChannelAttention

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("Loading model...")
model = keras.models.load_model(
    r"outputs_official\afcnn_official.h5",
    custom_objects={'ChannelAttention': ChannelAttention}
)
print("[OK] Model loaded!\n")

# ============================================================
# CHANGE THIS TO YOUR IMAGE PATH!
# ============================================================
image_path = r"C:\Users\chang\OneDrive\文件\AFCNN\dataset\disgust\disgust_00022.jpg"  # <-- Change this!

# Emotion names
emotions = ['anger', 'disgust', 'fear', 'happy', 'no emotion', 'sad', 'surprise']

# Load and preprocess image
print(f"Testing image: {image_path}")

# Use numpy to handle Unicode paths on Windows
import os
if os.path.exists(image_path):
    # Read image using numpy (handles Unicode paths)
    with open(image_path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    img = None

if img is None:
    print(f"ERROR: Cannot read image: {image_path}")
    print("Make sure the path is correct!")
    exit()

# Preprocess
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (80, 80))
img = img.astype(np.float32) / 255.0
img_batch = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img_batch, verbose=0)[0]
predicted_idx = np.argmax(predictions)
emotion = emotions[predicted_idx]
confidence = predictions[predicted_idx] * 100

# Show results
print("\n" + "="*50)
print("RESULT:")
print("="*50)
print(f"Predicted Emotion: {emotion.upper()}")
print(f"Confidence: {confidence:.2f}%")
print("\nAll Probabilities:")
print("-"*50)

for i, (em, prob) in enumerate(sorted(zip(emotions, predictions), key=lambda x: x[1], reverse=True), 1):
    bar = "#" * int(prob * 50)
    print(f"{i}. {em:12s}: {bar} {prob*100:5.2f}%")

print("="*50)
