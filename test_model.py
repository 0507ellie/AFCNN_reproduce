"""
Test Script for AFCNN Model
Test trained model on individual images or batch of images
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from model_official import ChannelAttention


class EmotionPredictor:
    """Predictor for emotion recognition"""

    def __init__(self, model_path, target_size=(80, 80)):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model
            target_size: Target image size
        """
        self.model_path = model_path
        self.target_size = target_size
        self.model = None

        self.emotion_names = [
            'Anger', 'Disgust', 'Fear', 'Happy',
            'No Emotion', 'Sad', 'Surprise'
        ]

        self.emotion_colors = {
            'Anger': 'red',
            'Disgust': 'green',
            'Fear': 'purple',
            'Happy': 'yellow',
            'No Emotion': 'gray',
            'Sad': 'blue',
            'Surprise': 'orange'
        }

    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from: {self.model_path}")
        custom_objects = {
            'ChannelAttention': ChannelAttention
        }
        self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
        print("Model loaded successfully!")
        return self

    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image
        """
        # Read image (handle Unicode paths on Windows)
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            img = None

        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(img, self.target_size)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def predict(self, image_path):
        """
        Predict emotion from image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Preprocess image
        img = self.preprocess_image(image_path)

        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)

        # Predict
        predictions = self.model.predict(img_batch, verbose=0)[0]

        # Get top prediction
        top_idx = np.argmax(predictions)
        top_emotion = self.emotion_names[top_idx]
        top_confidence = predictions[top_idx]

        # Create results dictionary
        results = {
            'emotion': top_emotion,
            'confidence': float(top_confidence),
            'probabilities': {
                emotion: float(prob)
                for emotion, prob in zip(self.emotion_names, predictions)
            }
        }

        return results

    def predict_batch(self, image_paths):
        """
        Predict emotions for multiple images

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction results
        """
        results = []

        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        return results

    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction results

        Args:
            image_path: Path to image
            save_path: Path to save visualization
        """
        # Get prediction
        results = self.predict(image_path)

        # Load original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Display image
        axes[0].imshow(img)
        axes[0].set_title(f"Predicted: {results['emotion']}\nConfidence: {results['confidence']:.2%}")
        axes[0].axis('off')

        # Display probability distribution
        emotions = list(results['probabilities'].keys())
        probabilities = list(results['probabilities'].values())

        colors = [self.emotion_colors.get(emotion, 'gray') for emotion in emotions]

        bars = axes[1].barh(emotions, probabilities, color=colors, edgecolor='black')
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Emotion Probabilities')
        axes[1].set_xlim([0, 1])

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            axes[1].text(
                width, bar.get_y() + bar.get_height() / 2,
                f' {width:.2%}',
                ha='left', va='center'
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_batch_predictions(self, image_paths, save_path=None, max_images=9):
        """
        Visualize predictions for multiple images

        Args:
            image_paths: List of image paths
            save_path: Path to save visualization
            max_images: Maximum number of images to display
        """
        num_images = min(len(image_paths), max_images)
        rows = int(np.ceil(np.sqrt(num_images)))
        cols = int(np.ceil(num_images / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten() if num_images > 1 else [axes]

        for idx, img_path in enumerate(image_paths[:max_images]):
            try:
                # Get prediction
                results = self.predict(img_path)

                # Load image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Display
                axes[idx].imshow(img)
                axes[idx].set_title(
                    f"{results['emotion']}\n{results['confidence']:.2%}",
                    fontsize=10
                )
                axes[idx].axis('off')

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Batch visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def test_on_webcam(model_path):
    """
    Test model on webcam feed (real-time)

    Args:
        model_path: Path to trained model
    """
    predictor = EmotionPredictor(model_path)
    predictor.load_model()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame temporarily
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            # Predict
            results = predictor.predict(temp_path)

            # Display results on frame
            text = f"{results['emotion']}: {results['confidence']:.2%}"
            cv2.putText(
                frame, text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )

        except Exception as e:
            print(f"Error: {e}")

        # Display frame
        cv2.imshow('Emotion Recognition', frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(temp_path):
        os.remove(temp_path)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test AFCNN Model')

    parser.add_argument(
        '--model',
        type=str,
        default=r"C:\Users\chang\OneDrive\文件\AFCNN\outputs\checkpoints\afcnn_best_model.h5",
        help='Path to trained model'
    )

    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to test image'
    )

    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Path to folder containing test images'
    )

    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Test on webcam'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save visualization'
    )

    args = parser.parse_args()

    # Create predictor
    predictor = EmotionPredictor(args.model)
    predictor.load_model()

    if args.webcam:
        # Test on webcam
        test_on_webcam(args.model)

    elif args.image:
        # Test on single image
        print(f"\nTesting on image: {args.image}")
        print("=" * 60)

        results = predictor.predict(args.image)

        print(f"\nPredicted Emotion: {results['emotion']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print("\nAll Probabilities:")
        for emotion, prob in results['probabilities'].items():
            print(f"  {emotion}: {prob:.2%}")

        # Visualize
        output_path = args.output or "prediction_result.png"
        predictor.visualize_prediction(args.image, output_path)

    elif args.folder:
        # Test on folder
        print(f"\nTesting on folder: {args.folder}")
        print("=" * 60)

        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if f.lower().endswith(image_extensions)
        ]

        print(f"Found {len(image_paths)} images")

        # Predict
        results = predictor.predict_batch(image_paths)

        # Print results
        for result in results:
            print(f"\n{os.path.basename(result['image_path'])}:")
            print(f"  Emotion: {result['emotion']}")
            print(f"  Confidence: {result['confidence']:.2%}")

        # Visualize
        output_path = args.output or "batch_prediction_results.png"
        predictor.visualize_batch_predictions(image_paths, output_path)

    else:
        print("Please specify --image, --folder, or --webcam")


if __name__ == "__main__":
    main()
