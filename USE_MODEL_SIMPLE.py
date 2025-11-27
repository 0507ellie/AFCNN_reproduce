"""
SIMPLE EXAMPLE: How to use the trained AFCNN model
Copy this code to use the model in your own projects
"""

import sys
import numpy as np
import cv2
from tensorflow import keras
from model_official import ChannelAttention

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# STEP 1: Load the trained model
# ============================================================
print("Loading trained model...")
model_path = r"outputs_official\afcnn_official.h5"

# IMPORTANT: Must include custom_objects for ChannelAttention layer
custom_objects = {'ChannelAttention': ChannelAttention}
model = keras.models.load_model(model_path, custom_objects=custom_objects)

print("[OK] Model loaded successfully!")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")
print()


# ============================================================
# STEP 2: Define emotion names
# ============================================================
EMOTION_NAMES = ['anger', 'disgust', 'fear', 'happy', 'no emotion', 'sad', 'surprise']


# ============================================================
# STEP 3: Create prediction function
# ============================================================
def predict_emotion(image_path):
    """
    Predict emotion from an image file

    Args:
        image_path: Path to image file (jpg, png, etc.)

    Returns:
        emotion: Predicted emotion name
        confidence: Confidence percentage (0-100)
        all_probs: Dictionary of all emotion probabilities
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (80, 80))             # Resize to 80x80
    img = img.astype(np.float32) / 255.0        # Normalize to [0,1]
    img_batch = np.expand_dims(img, axis=0)     # Add batch dimension

    # Predict
    predictions = model.predict(img_batch, verbose=0)[0]

    # Get results
    predicted_idx = np.argmax(predictions)
    emotion = EMOTION_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    # Create probability dictionary
    all_probs = {
        name: prob * 100
        for name, prob in zip(EMOTION_NAMES, predictions)
    }

    return emotion, confidence, all_probs


# ============================================================
# STEP 4: Example usage
# ============================================================
if __name__ == "__main__":
    # Example 1: Test on a single image
    print("=" * 60)
    print("EXAMPLE: Testing on a single image")
    print("=" * 60)

    # Change this to your image path
    test_image = r"dataset\happy\image_001.jpg"

    try:
        emotion, confidence, all_probs = predict_emotion(test_image)

        print(f"\nImage: {test_image}")
        print(f"\n[OK] Predicted Emotion: {emotion.upper()}")
        print(f"[OK] Confidence: {confidence:.2f}%")

        print(f"\nAll Probabilities:")
        for emotion_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(prob / 5)  # Bar chart
            print(f"  {emotion_name:12s}: {bar} {prob:5.2f}%")

    except Exception as e:
        print(f"Error: {e}")

    print()

    # Example 2: Test on multiple images
    print("=" * 60)
    print("EXAMPLE: Testing on multiple images")
    print("=" * 60)

    import os

    # Change this to your folder path
    test_folder = r"dataset\happy"

    if os.path.exists(test_folder):
        # Get all image files
        image_files = [
            f for f in os.listdir(test_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:5]  # Test first 5 images

        print(f"\nTesting {len(image_files)} images from {test_folder}\n")

        for img_file in image_files:
            img_path = os.path.join(test_folder, img_file)
            try:
                emotion, confidence, _ = predict_emotion(img_path)
                print(f"  {img_file:20s} → {emotion:12s} ({confidence:5.2f}%)")
            except Exception as e:
                print(f"  {img_file:20s} → Error: {e}")

    print()
    print("=" * 60)
    print("Done! You can now use the predict_emotion() function")
    print("in your own code by copying it from this file.")
    print("=" * 60)


# ============================================================
# HOW TO USE IN YOUR OWN CODE
# ============================================================
"""
1. Copy the predict_emotion() function to your code
2. Load the model once at the start:

   model = keras.models.load_model(
       "outputs_official/afcnn_official.h5",
       custom_objects={'ChannelAttention': ChannelAttention}
   )

3. Then call predict_emotion(image_path) whenever you need:

   emotion, confidence, all_probs = predict_emotion("my_image.jpg")
   print(f"Detected: {emotion} ({confidence:.1f}%)")

4. That's it!
"""
