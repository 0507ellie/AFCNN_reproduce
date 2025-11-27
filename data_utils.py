"""
Data Utilities for AFCNN Model
Handles data loading, preprocessing, and augmentation
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import matplotlib.pyplot as plt


class EmotionDataset:
    """Dataset handler for emotion recognition"""

    def __init__(self, data_dir, target_size=(80, 80), test_size=0.1, val_size=0.2):
        """
        Initialize the dataset

        Args:
            data_dir: Directory containing emotion folders
            target_size: Target image size (width, height)
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.test_size = test_size
        self.val_size = val_size

        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'no emotion': 4,
            'sad': 5,
            'surprise': 6
        }

        self.emotion_names = list(self.emotion_map.keys())
        self.num_classes = len(self.emotion_names)

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_data(self, balance_strategy='undersample', max_samples_per_class=1000):
        """
        Load all images and labels from the dataset with class balancing

        Args:
            balance_strategy: 'undersample', 'oversample', or 'none'
            max_samples_per_class: Maximum samples per class for undersampling
        """
        print("Loading dataset...")
        print("=" * 60)

        images = []
        labels = []

        # First pass: count samples per class
        class_counts = {}
        for emotion, label in self.emotion_map.items():
            emotion_dir = os.path.join(self.data_dir, emotion)
            if os.path.exists(emotion_dir):
                image_files = [f for f in os.listdir(emotion_dir)
                              if f.endswith(('.jpg', '.png', '.jpeg'))]
                class_counts[emotion] = len(image_files)

        # Determine target samples per class
        min_class_size = min(class_counts.values())

        if balance_strategy == 'undersample':
            # If max_samples_per_class > min class size, need to oversample minority classes
            # Otherwise just undersample to max_samples_per_class
            if max_samples_per_class > min_class_size:
                print(f"\nWARNING: max_samples_per_class ({max_samples_per_class}) > minimum class size ({min_class_size})")
                print(f"Switching to hybrid strategy: undersample majority + oversample minority to {max_samples_per_class}")
                target_samples = max_samples_per_class
                balance_strategy = 'hybrid'  # Will handle both under and oversampling
            else:
                target_samples = max_samples_per_class
                print(f"\nUndersampling to {target_samples} samples per class")
        elif balance_strategy == 'oversample':
            target_samples = max(class_counts.values())
            print(f"\nOversampling to {target_samples} samples per class")
        else:
            target_samples = None

        for emotion, label in self.emotion_map.items():
            emotion_dir = os.path.join(self.data_dir, emotion)

            if not os.path.exists(emotion_dir):
                print(f"Warning: {emotion_dir} does not exist. Skipping...")
                continue

            image_files = [f for f in os.listdir(emotion_dir)
                          if f.endswith(('.jpg', '.png', '.jpeg'))]

            print(f"{emotion}: {len(image_files)} images", end='')

            # Load images for this class
            class_images = []
            for img_file in image_files:
                img_path = os.path.join(emotion_dir, img_file)

                # Use numpy to handle Unicode paths on Windows
                try:
                    img_array = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    img = None

                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Resize if needed
                    if img.shape[:2] != self.target_size:
                        img = cv2.resize(img, self.target_size)

                    class_images.append(img)

            # Apply balancing strategy
            if balance_strategy != 'none' and target_samples is not None:
                if len(class_images) > target_samples:
                    # Undersample
                    indices = np.random.choice(len(class_images), target_samples, replace=False)
                    class_images = [class_images[i] for i in indices]
                elif len(class_images) < target_samples and (balance_strategy == 'oversample' or balance_strategy == 'hybrid'):
                    # Oversample with augmentation
                    original_size = len(class_images)
                    while len(class_images) < target_samples:
                        # Randomly duplicate and slightly augment
                        idx = np.random.randint(0, original_size)
                        img = class_images[idx].copy()

                        # Apply random augmentation
                        aug_type = np.random.randint(0, 4)
                        if aug_type == 0:
                            img = cv2.flip(img, 1)  # Horizontal flip
                        elif aug_type == 1:
                            # Slight rotation
                            angle = np.random.uniform(-5, 5)
                            h, w = img.shape[:2]
                            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                            img = cv2.warpAffine(img, M, (w, h))
                        elif aug_type == 2:
                            # Slight brightness adjustment
                            factor = np.random.uniform(0.9, 1.1)
                            img = np.clip(img * factor, 0, 255).astype(np.uint8)
                        # else: keep original

                        class_images.append(img)
                    class_images = class_images[:target_samples]

                print(f" -> balanced to {len(class_images)}")
            else:
                print()

            # Add to dataset
            images.extend(class_images)
            labels.extend([label] * len(class_images))

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels)

        print("=" * 60)
        print(f"Total images loaded: {len(images)}")
        print(f"Image shape: {images[0].shape}")

        return images, labels

    def preprocess_images(self, images):
        """
        Preprocess images: normalization

        Args:
            images: Array of images

        Returns:
            Preprocessed images
        """
        # Normalize to [0, 1]
        images = images / 255.0

        return images

    def split_data(self, images, labels):
        """
        Split data into train, validation, and test sets

        Args:
            images: Array of images
            labels: Array of labels

        Returns:
            Split datasets
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=self.test_size,
            random_state=42,
            stratify=labels
        )

        # Second split: train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=42,
            stratify=y_temp
        )

        # Convert labels to one-hot encoding
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        print("\nDataset split:")
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_data(self, balance_strategy='undersample', max_samples_per_class=600):
        """
        Prepare the complete dataset

        Args:
            balance_strategy: 'undersample', 'oversample', or 'none'
            max_samples_per_class: Maximum samples per class (default: 600 for balanced dataset)
        """
        # Load data with balancing
        images, labels = self.load_data(balance_strategy, max_samples_per_class)

        # Apply CLAHE (Local Contrast Enhancement) as mentioned in the paper
        # NOTE: apply_clahe already normalizes to [0, 1], so DON'T normalize again!
        print("\nApplying CLAHE (Local Contrast Enhancement)...")
        images = apply_clahe(images)

        # CLAHE already returns normalized images [0, 1], so skip preprocess_images()
        # images = self.preprocess_images(images)  # ← REMOVED: Would normalize twice!

        # Split
        self.X_train, self.X_val, self.X_test, \
        self.y_train, self.y_val, self.y_test = self.split_data(images, labels)

        return self

    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        y_train_labels = np.argmax(self.y_train, axis=1)
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        return dict(enumerate(class_weights))

    def create_data_augmentation(self):
        """
        Create data augmentation pipeline

        Implements augmentations suitable for micro-expression recognition.
        Important: Micro-expressions are subtle, so augmentations must be gentle.
        """
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),  # OK for facial expressions
            layers.RandomRotation(0.05),  # Reduced from 0.1 - subtle rotation only
            layers.RandomZoom(0.05),  # Reduced from 0.1 - minimal zoom
            layers.RandomBrightness(0.15),  # Reduced from 0.2 - gentle brightness changes
            layers.RandomContrast(0.15),  # Reduced from 0.2 - gentle contrast changes
            layers.RandomTranslation(0.05, 0.05),  # Small translations
        ], name='data_augmentation')

        return data_augmentation


def create_tf_dataset(X, y, batch_size=32, shuffle=True, augment=False, augmentation_layer=None):
    """
    Create a TensorFlow dataset

    Args:
        X: Images
        y: Labels
        batch_size: Batch size
        shuffle: Whether to shuffle
        augment: Whether to apply augmentation
        augmentation_layer: Augmentation layer to apply

    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)

    if augment and augmentation_layer is not None:
        dataset = dataset.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def apply_histogram_equalization(images):
    """
    Apply histogram equalization to images

    Args:
        images: Array of images

    Returns:
        Equalized images
    """
    equalized_images = []

    for img in images:
        # Convert to uint8 if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        # Convert to YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        # Equalize the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # Convert back to RGB
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        equalized_images.append(img_eq)

    return np.array(equalized_images, dtype=np.float32) / 255.0


def apply_clahe(images, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Implements Local Contrast Enhancement (LCE) as mentioned in the paper

    UPDATED: Changed clipLimit from 2.0 to 3.0 to match official paper code

    Args:
        images: Array of images
        clip_limit: Threshold for contrast limiting (paper uses 3.0)
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Enhanced images
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_images = []

    for img in images:
        # Convert to uint8 if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        # Convert to LAB color space
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])

        # Convert back to RGB
        img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

        enhanced_images.append(img_enhanced)

    return np.array(enhanced_images, dtype=np.float32) / 255.0


def visualize_samples(dataset, num_samples=10, save_path=None):
    """
    Visualize sample images from the dataset

    Args:
        dataset: EmotionDataset object
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    if dataset.X_train is None:
        print("Dataset not prepared. Call prepare_data() first.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(min(num_samples, len(dataset.X_train))):
        img = dataset.X_train[i]
        label_idx = np.argmax(dataset.y_train[i])
        label_name = dataset.emotion_names[label_idx]

        axes[i].imshow(img)
        axes[i].set_title(f"{label_name}")
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.close()


def print_dataset_statistics(dataset):
    """
    Print statistics about the dataset

    Args:
        dataset: EmotionDataset object
    """
    if dataset.y_train is None:
        print("Dataset not prepared. Call prepare_data() first.")
        return

    print("\nDataset Statistics:")
    print("=" * 60)

    for split_name, y_split in [('Train', dataset.y_train),
                                 ('Validation', dataset.y_val),
                                 ('Test', dataset.y_test)]:
        print(f"\n{split_name} Set:")
        labels = np.argmax(y_split, axis=1)

        for emotion_name, emotion_idx in dataset.emotion_map.items():
            count = np.sum(labels == emotion_idx)
            percentage = (count / len(labels)) * 100
            print(f"  {emotion_name}: {count} ({percentage:.1f}%)")

        print(f"  Total: {len(labels)}")


if __name__ == "__main__":
    # Test data loading
    data_dir = r"C:\Users\chang\OneDrive\文件\AFCNN\dataset"

    if os.path.exists(data_dir):
        dataset = EmotionDataset(data_dir, target_size=(80, 80))
        dataset.prepare_data()

        print_dataset_statistics(dataset)

        # Calculate class weights
        class_weights = dataset.get_class_weights()
        print("\nClass Weights:")
        for idx, weight in class_weights.items():
            print(f"  {dataset.emotion_names[idx]}: {weight:.3f}")

        # Visualize samples
        visualize_samples(dataset, num_samples=10,
                         save_path=r"C:\Users\chang\OneDrive\文件\AFCNN\sample_visualization.png")
    else:
        print(f"Dataset directory not found: {data_dir}")
        print("Please run extract_frames.py first to prepare the dataset.")
