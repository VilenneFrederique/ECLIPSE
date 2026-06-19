"""Keras layer building blocks for the ECLIPSE autoencoder.

Ported verbatim from the training code so the registered serialisable layers
reconstruct with exactly the same weight structure when loading published
weights.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable()
class PatchEmbedding(layers.Layer):
    """Convert a 1D spectrum into patch embeddings."""

    def __init__(self, embed_dim: int = 256, patch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.projection = layers.Dense(embed_dim)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.patch_size])
        return self.projection(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "patch_size": self.patch_size})
        return config


@keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Pre-norm transformer block."""

    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )

        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]
        )

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        x_norm = self.norm1(x)
        attn_out = self.att(x_norm, x_norm, training=training)
        x = x + attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm, training=training)
        x = x + ffn_out

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config
