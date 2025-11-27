"""
Evaluation Script for Official AFCNN Model
Tests the official trained model from the paper reproduction
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

from data_utils import EmotionDataset, create_tf_dataset
from model_official import ChannelAttention

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def evaluate_official_model(model_path, data_dir, output_dir):
    """
    Evaluate the official AFCNN model

    Args:
        model_path: Path to the trained official model
        data_dir: Path to dataset
        output_dir: Path to save evaluation results
    """
    print("="*60)
    print("OFFICIAL AFCNN MODEL EVALUATION")
    print("="*60)

    # Create output directory
    eval_dir = os.path.join(output_dir, "evaluation_official")
    os.makedirs(eval_dir, exist_ok=True)

    # Load model with custom objects
    print("\nLoading official model...")
    try:
        custom_objects = {
            'ChannelAttention': ChannelAttention
        }
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"[OK] Model loaded from: {model_path}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return

    # Prepare dataset (same as training)
    print("\nPreparing dataset...")
    dataset = EmotionDataset(
        data_dir=data_dir,
        target_size=(80, 80),
        test_size=0.1,
        val_size=0.2
    )

    dataset.prepare_data(
        balance_strategy='undersample',
        max_samples_per_class=600
    )

    # Create test dataset
    test_dataset = create_tf_dataset(
        dataset.X_test,
        dataset.y_test,
        batch_size=32,
        shuffle=False
    )

    # Get predictions
    print("\nGenerating predictions...")
    y_pred_probs = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(dataset.y_test, axis=1)

    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")

    # Compare with paper results (86.5%)
    paper_accuracy = 0.865
    improvement = (accuracy - paper_accuracy) * 100
    print(f"\n  Paper reported: 86.5%")
    print(f"  Difference: {improvement:+.2f}%")

    # Save overall metrics
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'paper_accuracy': paper_accuracy,
        'difference': float(improvement)
    }

    results_path = os.path.join(eval_dir, 'overall_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nOverall metrics saved to: {results_path}")

    # Classification report
    print("\n" + "="*60)
    print("Per-Class Metrics:")
    print("="*60)

    report = classification_report(
        y_true, y_pred,
        target_names=dataset.emotion_names,
        digits=4,
        zero_division=0
    )
    print(report)

    # Save classification report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=dataset.emotion_names,
        output_dict=True,
        zero_division=0
    )

    report_path = os.path.join(eval_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"\nClassification report saved to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=dataset.emotion_names,
        yticklabels=dataset.emotion_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Official AFCNN', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(eval_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=dataset.emotion_names,
        yticklabels=dataset.emotion_names,
        cbar_kws={'label': 'Percentage'}
    )
    plt.title('Normalized Confusion Matrix - Official AFCNN', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_norm_path = os.path.join(eval_dir, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    print(f"Normalized confusion matrix saved to: {cm_norm_path}")
    plt.close()

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(dataset.emotion_names, per_class_acc * 100, color='steelblue', alpha=0.8)
    plt.axhline(y=accuracy*100, color='red', linestyle='--',
                label=f'Overall Accuracy: {accuracy*100:.1f}%', linewidth=2)
    plt.title('Per-Class Accuracy - Official AFCNN', fontsize=16, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    acc_path = os.path.join(eval_dir, 'per_class_accuracy.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to: {acc_path}")
    plt.close()

    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print(f"All results saved to: {eval_dir}")
    print("="*60)

    return results


if __name__ == "__main__":
    # Configuration
    model_path = r"C:\Users\chang\OneDrive\文件\AFCNN\outputs_official\afcnn_official.h5"
    data_dir = r"C:\Users\chang\OneDrive\文件\AFCNN\dataset"
    output_dir = r"C:\Users\chang\OneDrive\文件\AFCNN\outputs_official"

    # Run evaluation
    results = evaluate_official_model(model_path, data_dir, output_dir)
