"""
Official Training Script - Matching Paper's GitHub Code Exactly
This implements the training procedure from the official repository
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, CSVLogger
from datetime import datetime
import matplotlib.pyplot as plt

# Import official model
from model_official import build_afcnn_model_official, cosine_annealing_schedule
from data_utils import EmotionDataset, print_dataset_statistics


class OfficialTrainingConfig:
    """Configuration matching the paper's official code"""

    def __init__(self):
        # Paths
        self.data_dir = r"C:\Users\chang\OneDrive\文件\AFCNN\dataset"
        self.output_dir = r"C:\Users\chang\OneDrive\文件\AFCNN\outputs_official"
        os.makedirs(self.output_dir, exist_ok=True)

        # Model parameters (from paper)
        self.input_shape = (80, 80, 3)
        self.num_classes = 7

        # Training parameters (exactly from paper's code)
        self.batch_size = 32
        self.epochs = 150
        self.initial_lr = 0.001

        # Data parameters
        self.balance_strategy = 'undersample'
        self.max_samples_per_class = 600

        # Early stopping (from paper)
        self.patience = 15

    def save(self, filepath):
        """Save configuration"""
        config_dict = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'initial_lr': self.initial_lr,
            'balance_strategy': self.balance_strategy,
            'max_samples_per_class': self.max_samples_per_class,
            'patience': self.patience,
            'architecture': 'Official AFCNN from paper',
            'channel_attention': '3 modules (after each pooling)',
            'clahe_clip_limit': 3.0
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)


class OfficialTrainer:
    """Trainer using official paper implementation"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        self.dataset = None

    def prepare_dataset(self):
        """Load and prepare dataset"""
        print("Preparing dataset (Official Paper Method)...")
        print("=" * 70)

        self.dataset = EmotionDataset(
            data_dir=self.config.data_dir,
            target_size=self.config.input_shape[:2],
            test_size=0.1,  # 10% for test
            val_size=0.2     # 20% of remaining for validation
        )

        # Prepare with balancing and CLAHE (clipLimit=3.0)
        self.dataset.prepare_data(
            balance_strategy=self.config.balance_strategy,
            max_samples_per_class=self.config.max_samples_per_class
        )

        print_dataset_statistics(self.dataset)
        return self

    def build_model(self):
        """Build the official AFCNN model"""
        print("\nBuilding Official AFCNN Model...")
        print("=" * 70)
        print("Architecture: 3 Channel Attention modules (after each pooling)")
        print("CLAHE: clipLimit=3.0 (stronger contrast)")
        print()

        self.model = build_afcnn_model_official(
            input_shape=self.config.input_shape,
            num_classes=self.config.num_classes
        )

        # Compile with simple Adam optimizer (no gradient clipping)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.initial_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.summary()
        return self

    def get_callbacks(self):
        """Create callbacks matching paper"""
        callbacks = []

        # Early Stopping (monitors val_loss, patience=15)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Cosine Annealing Learning Rate Schedule
        def lr_schedule(epoch, lr):
            return cosine_annealing_schedule(epoch, lr, self.config.epochs)

        lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
        callbacks.append(lr_scheduler)

        # CSV Logger
        csv_path = os.path.join(self.config.output_dir, "training_log.csv")
        csv_logger = CSVLogger(csv_path)
        callbacks.append(csv_logger)

        return callbacks

    def train(self):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built")
        if self.dataset is None:
            raise ValueError("Dataset not prepared")

        print("\nStarting Training (Official Paper Method)...")
        print("=" * 70)
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Initial LR: {self.config.initial_lr}")
        print(f"LR Schedule: Cosine Annealing")
        print(f"Early Stopping: Patience {self.config.patience}")
        print()

        # Get callbacks
        callbacks = self.get_callbacks()

        # Train (NO data augmentation during training in paper's code)
        self.history = self.model.fit(
            self.dataset.X_train,
            self.dataset.y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=(self.dataset.X_val, self.dataset.y_val),
            callbacks=callbacks,
            verbose=1
        )

        print("\nTraining completed!")
        return self

    def evaluate(self):
        """Evaluate on test set"""
        if self.model is None:
            raise ValueError("Model not trained")

        print("\nEvaluating on test set...")
        print("=" * 70)

        # Evaluate
        test_loss, test_acc = self.model.evaluate(
            self.dataset.X_test,
            self.dataset.y_test,
            verbose=0
        )

        # Get predictions for detailed metrics
        y_pred = self.model.predict(self.dataset.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.dataset.y_test, axis=1)

        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

        print(f"\nTest Results:")
        print(f"  Accuracy:  {test_acc*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1-Score:  {f1*100:.2f}%")

        # Save results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_score_macro': float(f1)
        }

        results_path = os.path.join(self.config.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to: {results_path}")
        return results

    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            return

        history = self.history.history

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(history['accuracy'], label='Training')
        axes[0].plot(history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history['loss'], label='Training')
        axes[1].plot(history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.config.output_dir, 'training_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
        plt.close()

    def save_model(self):
        """Save trained model"""
        if self.model is None:
            return

        model_path = os.path.join(self.config.output_dir, 'afcnn_official.h5')
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")


def main():
    """Main training function"""
    print("=" * 70)
    print("AFCNN OFFICIAL TRAINING - Matching Paper's GitHub Code")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create config
    config = OfficialTrainingConfig()

    # Save config
    config_path = os.path.join(config.output_dir, 'training_config.json')
    config.save(config_path)
    print(f"Configuration saved to: {config_path}\n")

    # Create trainer
    trainer = OfficialTrainer(config)

    # Prepare dataset
    trainer.prepare_dataset()

    # Build model
    trainer.build_model()

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()

    # Plot history
    trainer.plot_history()

    # Save model
    trainer.save_model()

    print("\n" + "=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Training completed successfully!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Run training
    results = main()

    print("\n" + "=" * 70)
    print("Next step: Run evaluation with:")
    print("  python evaluate_improved.py")
    print("=" * 70)
