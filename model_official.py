"""
AFCNN Model Implementation - OFFICIAL VERSION FROM PAPER
Based on the exact code from the paper's GitHub repository
https://github.com/C3R8U/afcnn-emotion-recognition

KEY DIFFERENCES FROM OUR PREVIOUS VERSION:
1. Channel Attention uses units=1 (scalar) not units=channels
2. Channel Attention applied AFTER EACH pooling layer (3 times total)
3. CLAHE clipLimit=3.0 (not 2.0)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class ChannelAttention(layers.Layer):
    """
    Channel Attention Module - OFFICIAL PAPER VERSION

    IMPORTANT: This uses Dense(1) not Dense(channels)!
    The attention weight is a SCALAR, not per-channel weights.
    """

    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.global_avg = layers.GlobalAveragePooling2D()
        # CRITICAL: units=1, not units=channels!
        self.dense1 = layers.Dense(units=1, activation='relu')
        self.dense2 = layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        # Squeeze operation: [batch, H, W, C] → [batch, C]
        x = self.global_avg(inputs)

        # Expand dims: [batch, C] → [batch, 1, 1, C]
        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=1)

        # Excitation operation: [batch, 1, 1, C] → [batch, 1, 1, 1]
        x = self.dense1(x)
        attention_weights = self.dense2(x)

        # Scale features: broadcast scalar attention to all channels
        return inputs * attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


def build_afcnn_model_official(input_shape=(80, 80, 3), num_classes=7):
    """
    Build AFCNN Model - EXACT ARCHITECTURE FROM PAPER

    Key Features:
    - 3 Convolutional Blocks (32, 64, 128 filters)
    - Channel Attention after EACH MaxPooling layer (3 attention modules)
    - BatchNorm after each Conv2D
    - Dropout(0.5) before final classification

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)

    # ============================================================
    # Convolutional Block 1
    # ============================================================
    x = layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Channel Attention 1 (AFTER POOLING!)
    x = ChannelAttention()(x)

    # ============================================================
    # Convolutional Block 2
    # ============================================================
    x = layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Channel Attention 2 (AFTER POOLING!)
    x = ChannelAttention()(x)

    # ============================================================
    # Convolutional Block 3
    # ============================================================
    x = layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Channel Attention 3 (AFTER POOLING!)
    x = ChannelAttention()(x)

    # ============================================================
    # Classification Block
    # ============================================================
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='AFCNN_Official')
    return model


def compile_model_official(model, initial_lr=0.001):
    """
    Compile model with Adam optimizer and categorical crossentropy

    Note: Paper uses simple Adam optimizer, no learning rate schedule in compile
    The learning rate schedule is applied via callback during training
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def cosine_annealing_schedule(epoch, lr, epochs_total=150):
    """
    Cosine annealing learning rate schedule
    Exactly as implemented in the paper

    Args:
        epoch: Current epoch
        lr: Current learning rate
        epochs_total: Total number of epochs

    Returns:
        New learning rate
    """
    import numpy as np
    cos_inner = np.pi * (epoch % epochs_total)
    cos_inner /= epochs_total
    cos_out = np.cos(cos_inner) + 1
    return float(lr * cos_out / 2)


if __name__ == "__main__":
    # Test model creation
    print("="*70)
    print("Creating AFCNN Model - OFFICIAL PAPER VERSION")
    print("="*70)

    model = build_afcnn_model_official(input_shape=(80, 80, 3), num_classes=7)
    model.summary()

    print("\n" + "="*70)
    print("Key Differences from Previous Implementation:")
    print("="*70)
    print("1. Channel Attention uses Dense(1) - scalar attention weight")
    print("2. Channel Attention applied 3 times (after each pooling layer)")
    print("3. No gradient clipping in optimizer (just vanilla Adam)")
    print("4. Cosine annealing via callback, not in optimizer")
    print("="*70)

    # Show model parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024*1024):.2f} MB")
