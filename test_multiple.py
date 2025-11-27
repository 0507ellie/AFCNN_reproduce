"""
Test Multiple Images - See model predictions across different emotions
"""

import sys
import numpy as np
import cv2
from tensorflow import keras
from model_official import ChannelAttention
import os

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("Loading model...")
model = keras.models.load_model(
    r"outputs_official\afcnn_official.h5",
    custom_objects={'ChannelAttention': ChannelAttention}
)
print("[OK] Model loaded!\n")

# Emotion names
emotions = ['anger', 'disgust', 'fear', 'happy', 'no emotion', 'sad', 'surprise']

# Test images from each emotion folder
test_images = [
    r"dataset\anger\anger_00005.jpg",
    r"dataset\disgust\disgust_00022.jpg",
    r"dataset\fear\fear_00005.jpg",
    r"dataset\happy\happy_00005.jpg",
    r"dataset\no emotion\noemo_00005.jpg",
    r"dataset\sad\sad_00005.jpg",
    r"dataset\surprise\surprise_00005.jpg",
]

def predict_image(image_path):
    """Predict emotion from image"""
    if not os.path.exists(image_path):
        return None, None, None

    # Read image using numpy (handles Unicode paths)
    with open(image_path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None, None, None

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

    return emotion, confidence, predictions

print("="*70)
print("TESTING MODEL ON DIFFERENT EMOTIONS")
print("="*70)
print()

results = []
for img_path in test_images:
    # Get true label from folder name
    true_emotion = img_path.split('\\')[1].replace('_', ' ')

    emotion, confidence, probs = predict_image(img_path)

    if emotion is None:
        print(f"[SKIP] {true_emotion:12s} - File not found")
        continue

    # Check if correct
    match = "[OK]" if emotion.lower().replace(' ', '') == true_emotion.lower().replace(' ', '') else "[X]"

    print(f"{match} True: {true_emotion:12s} | Predicted: {emotion:12s} ({confidence:5.1f}%)")
    results.append((true_emotion, emotion, confidence, probs))

print()
print("="*70)
print("SUMMARY")
print("="*70)

correct = sum(1 for true, pred, _, _ in results if true.lower().replace(' ', '') == pred.lower().replace(' ', ''))
total = len(results)
accuracy = (correct / total * 100) if total > 0 else 0

print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
print()

# Show confusion examples
print("Most Confident Predictions:")
print("-"*70)
sorted_results = sorted(results, key=lambda x: x[2], reverse=True)[:3]
for true, pred, conf, probs in sorted_results:
    match = "[OK]" if true.lower().replace(' ', '') == pred.lower().replace(' ', '') else "[X]"
    print(f"{match} {true:12s} -> {pred:12s} ({conf:5.1f}%)")

print()
print("Least Confident Predictions:")
print("-"*70)
sorted_results = sorted(results, key=lambda x: x[2])[:3]
for true, pred, conf, probs in sorted_results:
    match = "[OK]" if true.lower().replace(' ', '') == pred.lower().replace(' ', '') else "[X]"
    print(f"{match} {true:12s} -> {pred:12s} ({conf:5.1f}%)")

print()
print("="*70)
