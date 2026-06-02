"""
SpecCheck - Deep Learning for MS/MS Spectrum Analysis
======================================================

Approaches:
1. Standard autoencoder for spectrum reconstruction
2. Conditional autoencoder with precursor m/z, charge, ion mobility
3. Clustering in latent space for dark proteome analysis

Usage:
------
# Train CONDITIONAL autoencoder (recommended for clustering)
python SpecCheckVSC.py train-conditional-ae -i /path/to/parquet -o /path/to/output

# Cluster unidentified spectra (dark proteome analysis)
python SpecCheckVSC.py cluster -i /path/to/parquet -a /path/to/autoencoder -o /path/to/output

# Visualize top clusters
python SpecCheckVSC.py visualize-clusters -i /path/to/parquet -c /path/to/clustering -o /path/to/output

# Diagnose autoencoder quality
python SpecCheckVSC.py diagnose -i /path/to/parquet -a /path/to/autoencoder -o /path/to/output

Note: For clustering by peptide identity, use the conditional autoencoder.
The conditioning (precursor_mz, charge, IM) helps the latent space encode
peptide-specific patterns, so same-peptide spectra cluster together.
"""

import os
import gc
import glob
import random
import math
import time
import json
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Dict
import resource
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import roc_auc_score, average_precision_score


# ================================
# Environment Configuration
# ================================

def configure_environment(seed: int = 42, prefer_mixed_precision: bool = True):
    """Configure TensorFlow environment for optimal performance."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except:
        pass

    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.set_memory_growth(gpu, True)
                except Exception:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception as e:
                        print(f"Warning: could not set memory growth for GPU {gpu}: {e}")
            print(f"Detected {len(gpus)} GPU(s). Memory growth enabled.")
        else:
            print("No GPUs detected; running on CPU.")
    except Exception as e:
        print(f"GPU configuration failed: {e}")

    if prefer_mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision (float16) enabled.")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")
    
    tf.keras.backend.clear_session()
    gc.collect()


# ================================
# Configuration
# ================================

class Config:
    """Centralized configuration."""
    SEED = 42
    
    # Binning parameters (for autoencoder input)
    MZ_MIN = 100.0
    MZ_MAX = 1700.0
    BIN_SIZE = 0.5
    N_BINS = int((MZ_MAX - MZ_MIN) / BIN_SIZE)  # 3200
    
    # Preprocessing
    RELATIVE_INTENSITY_THRESHOLD = 0.01
    TOP_N_PEAKS = 100  # Keep only top N most intense peaks (set to None to disable)
    
    # Ion mobility normalization (1/K₀ range for peptides)
    IM_MIN = 0.6   # Vs/cm² - adjust based on your data
    IM_MAX = 1.6   # Vs/cm² - adjust based on your data
    
    # Precursor features
    MAX_CHARGE = 6
    PRECURSOR_MZ_MAX = 1700.0
    
    # Autoencoder architecture
    LATENT_DIM = 256           # Dimension of latent space
    AE_EMBED_DIM = 256         # Transformer embedding dimension
    AE_NUM_HEADS = 8
    AE_NUM_LAYERS = 4
    AE_FF_DIM = 512
    AE_PATCH_SIZE = 16
    AE_DROPOUT = 0.1
    
    # Diffusion parameters
    TIMESTEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02
    SCHEDULE = 'cosine'
    
    # Diffusion model architecture (operates in latent space)
    DIFF_EMBED_DIM = 256
    DIFF_NUM_HEADS = 8
    DIFF_NUM_LAYERS = 6
    DIFF_FF_DIM = 512
    DIFF_DROPOUT = 0.1
    
    # Conditioning embedding
    TIME_EMBED_DIM = 256
    COND_EMBED_DIM = 256
    
    # Training - Autoencoder
    AE_BATCH_SIZE = 256
    AE_INITIAL_LR = 1e-4
    AE_EPOCHS = 50
    AE_WARMUP_EPOCHS = 5
    
    # Training - Diffusion
    DIFF_BATCH_SIZE = 256
    DIFF_INITIAL_LR = 1e-4
    DIFF_EPOCHS = 100
    DIFF_WARMUP_EPOCHS = 5
    
    WEIGHT_DECAY = 1e-5
    
    # Scoring
    SCORE_TIMESTEPS = [250, 500, 750]  # Single timestep for faster scoring (was [250, 500, 750])


# ================================
# Preprocessing (same as before)
# ================================

@tf.function
def bin_spectrum_tf(mz, intensity, config=Config):
    """Convert raw peaks to binned representation with peak filtering."""
    mz = tf.cast(mz, tf.float32)
    intensity = tf.cast(intensity, tf.float32)
    
    mask = (mz >= config.MZ_MIN) & (mz < config.MZ_MAX) & (intensity > 0)
    mz = tf.boolean_mask(mz, mask)
    intensity = tf.boolean_mask(intensity, mask)
    
    max_int = tf.reduce_max(intensity)
    intensity = tf.math.divide_no_nan(intensity, max_int)
    
    mask = intensity >= config.RELATIVE_INTENSITY_THRESHOLD
    mz = tf.boolean_mask(mz, mask)
    intensity = tf.boolean_mask(intensity, mask)
    
    # Keep only top N peaks (if configured)
    if config.TOP_N_PEAKS is not None:
        n_peaks = tf.shape(intensity)[0]
        k = tf.minimum(n_peaks, config.TOP_N_PEAKS)
        # Get indices of top k peaks by intensity
        _, top_indices = tf.math.top_k(intensity, k=k)
        mz = tf.gather(mz, top_indices)
        intensity = tf.gather(intensity, top_indices)
    
    intensity = tf.sqrt(intensity)
    
    bin_indices = tf.cast((mz - config.MZ_MIN) / config.BIN_SIZE, tf.int32)
    bin_indices = tf.clip_by_value(bin_indices, 0, config.N_BINS - 1)
    
    binned = tf.zeros(config.N_BINS, dtype=tf.float32)
    binned = tf.tensor_scatter_nd_max(
        binned, 
        tf.expand_dims(bin_indices, 1), 
        intensity
    )
    
    max_val = tf.reduce_max(binned)
    binned = tf.math.divide_no_nan(binned, max_val)
    
    return binned


@tf.function
def preprocess_spectrum_with_im(mz, intensity, precursor_mz, precursor_charge, 
                                 ion_mobility, config=Config):
    """Preprocess spectrum with ion mobility conditioning."""
    binned = bin_spectrum_tf(mz, intensity, config)
    
    charge_int = tf.cast(precursor_charge, tf.int32)
    charge_int = tf.clip_by_value(charge_int - 1, 0, config.MAX_CHARGE - 1)
    charge_onehot = tf.one_hot(charge_int, config.MAX_CHARGE)
    
    precursor_mz_norm = tf.cast(precursor_mz, tf.float32) / config.PRECURSOR_MZ_MAX
    
    ion_mobility = tf.cast(ion_mobility, tf.float32)
    im_norm = (ion_mobility - config.IM_MIN) / (config.IM_MAX - config.IM_MIN)
    im_norm = tf.clip_by_value(im_norm, 0.0, 1.0)
    
    conditioning = tf.concat([charge_onehot, [precursor_mz_norm], [im_norm]], axis=0)
    
    return binned, conditioning


# ================================
# Autoencoder Components
# ================================

@keras.utils.register_keras_serializable()
class PatchEmbedding(layers.Layer):
    """Convert 1D spectrum into patch embeddings."""
    
    def __init__(self, embed_dim: int = 256, patch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.projection = layers.Dense(embed_dim)
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        # x: [batch, n_bins] -> [batch, num_patches, patch_size]
        x = tf.reshape(x, [batch_size, -1, self.patch_size])
        return self.projection(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'patch_size': self.patch_size})
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
            dropout=dropout
        )
        
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])

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
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate,
        })
        return config
    
# ================================
# Conditional Autoencoder (with precursor m/z, charge, ion mobility)
# ================================

@keras.utils.register_keras_serializable()
class ConditionalSpectrumEncoder(keras.Model):
    """
    Encode binned spectrum + conditioning to latent vector.
    
    Conditioning includes: precursor_mz, charge (one-hot), ion_mobility
    The conditioning is projected and added as a token.
    """
    
    def __init__(
        self,
        n_bins: int = 3200,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        latent_dim: int = 256,
        cond_dim: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_bins = n_bins
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.num_patches = n_bins // patch_size
        
        self.patch_embed = PatchEmbedding(embed_dim, patch_size)
        
        self.cond_proj = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(embed_dim)
        ], name='cond_projection')
        
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
        
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(1, self.num_patches + 2, embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        
        self.to_latent = keras.Sequential([
            layers.Dense(latent_dim, activation='gelu'),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(latent_dim)
        ])
    
    def call(self, inputs, training=False):
        x, cond = inputs
        batch_size = tf.shape(x)[0]
        
        x = self.patch_embed(x)
        
        cond_token = self.cond_proj(cond)
        cond_token = tf.expand_dims(cond_token, 1)
        
        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        
        x = tf.concat([cls_tokens, cond_token, x], axis=1)
        x = x + self.pos_embed
        
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        x = self.final_norm(x)
        cls_output = x[:, 0, :]
        z = self.to_latent(cls_output)
        
        return z
    
    def get_config(self):
        return {
            'n_bins': self.n_bins,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'cond_dim': self.cond_dim,
        }


@keras.utils.register_keras_serializable()
class ConditionalSpectrumDecoder(keras.Model):
    """
    Two-head conditional decoder: Latent + Conditioning -> Spectrum
    """
    
    def __init__(
        self,
        n_bins: int = 3200,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        latent_dim: int = 256,
        cond_dim: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_bins = n_bins
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.num_patches = n_bins // patch_size
        
        self.cond_proj = keras.Sequential([
            layers.Dense(embed_dim, activation='gelu'),
            layers.Dense(embed_dim)
        ], name='cond_projection')
        
        # Input is latent + cond concatenated
        self.from_latent = keras.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(embed_dim * self.num_patches),
            layers.Reshape((self.num_patches, embed_dim))
        ])
        
        self.pos_embed = self.add_weight(
            name='dec_pos_embed',
            shape=(1, self.num_patches + 1, embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        
        self.presence_head = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(patch_size, dtype='float32')
        ], name='presence_head')
        
        self.intensity_head = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(patch_size, dtype='float32')
        ], name='intensity_head')
        
        self.presence_threshold = 0.5
        self.presence_temperature = 2.0
    
    def call(self, inputs, training=False):
        z, cond = inputs
        batch_size = tf.shape(z)[0]
        
        z_cond = tf.concat([z, cond], axis=-1)
        x = self.from_latent(z_cond)
        
        cond_token = self.cond_proj(cond)
        cond_token = tf.expand_dims(cond_token, 1)
        
        x = tf.concat([cond_token, x], axis=1)
        x = x + self.pos_embed
        
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        x = self.final_norm(x)
        x = x[:, 1:, :]
        
        presence_logits = self.presence_head(x)
        presence_logits = tf.reshape(presence_logits, [batch_size, self.n_bins])
        presence_prob = tf.nn.sigmoid(presence_logits * self.presence_temperature)
        
        intensity_raw = self.intensity_head(x)
        intensity_raw = tf.reshape(intensity_raw, [batch_size, self.n_bins])
        intensity = tf.nn.sigmoid(intensity_raw)
        
        self.last_presence_prob = presence_prob
        self.last_presence_logits = presence_logits
        self.last_intensity = intensity
        
        if training:
            x_recon = presence_prob * intensity
        else:
            presence_mask = tf.cast(presence_prob > self.presence_threshold, tf.float32)
            x_recon = presence_mask * intensity
        
        return x_recon
    
    def get_config(self):
        return {
            'n_bins': self.n_bins,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'cond_dim': self.cond_dim,
        }


@keras.utils.register_keras_serializable()
class ConditionalSpectrumAutoencoder(keras.Model):
    """
    Conditional Autoencoder: (Spectrum, Conditioning) -> Latent -> Spectrum
    
    Conditioning includes precursor_mz, charge, ion_mobility.
    The latent space encodes spectral patterns CONDITIONED on precursor info,
    making same-peptide spectra cluster together.
    """
    
    def __init__(
        self,
        n_bins: int = 3200,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        latent_dim: int = 256,
        cond_dim: int = 8,
        dropout: float = 0.1,
        use_kl: bool = False,
        kl_weight: float = 1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.use_kl = use_kl
        self.kl_weight = kl_weight
        
        self.encoder = ConditionalSpectrumEncoder(
            n_bins=n_bins,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            latent_dim=latent_dim if not use_kl else latent_dim * 2,
            cond_dim=cond_dim,
            dropout=dropout,
        )
        
        self.decoder = ConditionalSpectrumDecoder(
            n_bins=n_bins,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            dropout=dropout,
        )
        
        self.recon_loss_tracker = keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.cosine_sim_tracker = keras.metrics.Mean(name='cosine_sim')
        self.sparsity_tracker = keras.metrics.Mean(name='sparsity')
        self.presence_acc_tracker = keras.metrics.Mean(name='presence_acc')
        self.fp_rate_tracker = keras.metrics.Mean(name='fp_rate')
    
    def encode(self, x, cond, training=False):
        z = self.encoder((x, cond), training=training)
        
        if self.use_kl:
            mu = z[:, :self.latent_dim]
            logvar = z[:, self.latent_dim:]
            
            if training:
                std = tf.exp(0.5 * logvar)
                eps = tf.random.normal(tf.shape(std))
                z = mu + eps * std
            else:
                z = mu
            
            return z, mu, logvar
        
        return z
    
    def decode(self, z, cond, training=False):
        return self.decoder((z, cond), training=training)
    
    def call(self, inputs, training=False):
        x, cond = inputs
        
        if self.use_kl:
            z, mu, logvar = self.encode(x, cond, training=training)
        else:
            z = self.encode(x, cond, training=training)
        
        x_recon = self.decode(z, cond, training=training)
        return x_recon
    
    def train_step(self, data):
        x, cond = data
        
        with tf.GradientTape() as tape:
            if self.use_kl:
                z, mu, logvar = self.encode(x, cond, training=True)
                x_recon = self.decode(z, cond, training=True)
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + logvar - tf.square(mu) - tf.exp(logvar)
                )
            else:
                z = self.encode(x, cond, training=True)
                x_recon = self.decode(z, cond, training=True)
                kl_loss = 0.0
            
            presence_prob = self.decoder.last_presence_prob
            presence_logits = self.decoder.last_presence_logits
            intensity = self.decoder.last_intensity
            
            peak_mask = tf.cast(x > 0.05, tf.float32)
            
            presence_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=peak_mask, logits=presence_logits
            )
            presence_loss = tf.reduce_mean(presence_bce)
            
            intensity_error = tf.square(x - intensity)
            masked_intensity_error = intensity_error * peak_mask
            num_peaks = tf.reduce_sum(peak_mask, axis=-1, keepdims=True) + 1e-6
            intensity_loss = tf.reduce_mean(
                tf.reduce_sum(masked_intensity_error, axis=-1) / tf.squeeze(num_peaks)
            )
            
            x_norm = tf.nn.l2_normalize(x, axis=-1)
            x_recon_norm = tf.nn.l2_normalize(x_recon, axis=-1)
            cos_sim = tf.reduce_sum(x_norm * x_recon_norm, axis=-1)
            spectral_angle_loss = tf.reduce_mean(1 - cos_sim)
            
            false_positive_mask = (1 - peak_mask)
            false_positive_penalty = tf.reduce_mean(presence_prob * false_positive_mask)
            
            recon_loss = (
                1.0 * presence_loss +
                1.0 * intensity_loss +
                0.5 * spectral_angle_loss +
                0.5 * false_positive_penalty
            )
            
            total_loss = recon_loss
            if self.use_kl:
                total_loss += self.kl_weight * kl_loss
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.cosine_sim_tracker.update_state(tf.reduce_mean(cos_sim))
        
        sparsity = tf.reduce_mean(tf.cast(presence_prob < 0.1, tf.float32))
        self.sparsity_tracker.update_state(sparsity)
        
        presence_pred = tf.cast(presence_prob > 0.5, tf.float32)
        presence_acc = tf.reduce_mean(tf.cast(tf.equal(presence_pred, peak_mask), tf.float32))
        self.presence_acc_tracker.update_state(presence_acc)
        
        predicted_peaks = tf.reduce_sum(presence_pred)
        false_positives = tf.reduce_sum(presence_pred * (1 - peak_mask))
        fp_rate = false_positives / (predicted_peaks + 1e-6)
        self.fp_rate_tracker.update_state(fp_rate)
        
        return {
            'loss': self.total_loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'cosine_sim': self.cosine_sim_tracker.result(),
            'sparsity': self.sparsity_tracker.result(),
            'presence_acc': self.presence_acc_tracker.result(),
            'fp_rate': self.fp_rate_tracker.result(),
        }
    
    def test_step(self, data):
        x, cond = data
        
        if self.use_kl:
            z, mu, logvar = self.encode(x, cond, training=False)
            x_recon = self.decode(z, cond, training=False)
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        else:
            z = self.encode(x, cond, training=False)
            x_recon = self.decode(z, cond, training=False)
            kl_loss = 0.0
        
        presence_prob = self.decoder.last_presence_prob
        presence_logits = self.decoder.last_presence_logits
        intensity = self.decoder.last_intensity
        
        peak_mask = tf.cast(x > 0.05, tf.float32)
        
        presence_bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=peak_mask, logits=presence_logits
        )
        presence_loss = tf.reduce_mean(presence_bce)
        
        intensity_error = tf.square(x - intensity)
        masked_intensity_error = intensity_error * peak_mask
        num_peaks = tf.reduce_sum(peak_mask, axis=-1, keepdims=True) + 1e-6
        intensity_loss = tf.reduce_mean(
            tf.reduce_sum(masked_intensity_error, axis=-1) / tf.squeeze(num_peaks)
        )
        
        x_norm = tf.nn.l2_normalize(x, axis=-1)
        x_recon_norm = tf.nn.l2_normalize(x_recon, axis=-1)
        cos_sim = tf.reduce_sum(x_norm * x_recon_norm, axis=-1)
        spectral_angle_loss = tf.reduce_mean(1 - cos_sim)
        
        false_positive_mask = (1 - peak_mask)
        false_positive_penalty = tf.reduce_mean(presence_prob * false_positive_mask)
        
        recon_loss = (
            1.0 * presence_loss +
            1.0 * intensity_loss +
            0.5 * spectral_angle_loss +
            0.5 * false_positive_penalty
        )
        
        total_loss = recon_loss
        if self.use_kl:
            total_loss += self.kl_weight * kl_loss
        
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.cosine_sim_tracker.update_state(tf.reduce_mean(cos_sim))
        
        sparsity = tf.reduce_mean(tf.cast(presence_prob < 0.1, tf.float32))
        self.sparsity_tracker.update_state(sparsity)
        
        presence_pred = tf.cast(presence_prob > 0.5, tf.float32)
        presence_acc = tf.reduce_mean(tf.cast(tf.equal(presence_pred, peak_mask), tf.float32))
        self.presence_acc_tracker.update_state(presence_acc)
        
        predicted_peaks = tf.reduce_sum(presence_pred)
        false_positives = tf.reduce_sum(presence_pred * (1 - peak_mask))
        fp_rate = false_positives / (predicted_peaks + 1e-6)
        self.fp_rate_tracker.update_state(fp_rate)
        
        return {
            'loss': self.total_loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'cosine_sim': self.cosine_sim_tracker.result(),
            'sparsity': self.sparsity_tracker.result(),
            'presence_acc': self.presence_acc_tracker.result(),
            'fp_rate': self.fp_rate_tracker.result(),
        }
    
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker,
                self.cosine_sim_tracker, self.sparsity_tracker,
                self.presence_acc_tracker, self.fp_rate_tracker]
    
    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'cond_dim': self.cond_dim,
            'use_kl': self.use_kl,
            'kl_weight': self.kl_weight,
        }

# ================================
# Dataset Creation
# ================================
def create_dataset(file_list: List[str], batch_size: int, config: Config,
                             identified_only: bool = True):
    """Create dataset for training (spectra + conditioning)."""
    
    def generator():
        for filepath in file_list:
            df = pd.read_parquet(filepath)
            if identified_only:
                df = df[df['Outcome'] == 1]
            
            for _, row in df.iterrows():
                mz = np.array(row['mz_array'], dtype=np.float32)
                intensity = np.array(row['intensity_array'], dtype=np.float32)
                precursor_mz = float(row['precursor_mz'])
                precursor_charge = float(row['precursor_charge'])
                ion_mobility = float(row['IM'])
                
                if np.isnan(ion_mobility) or ion_mobility <= 0:
                    continue
                
                spectrum, conditioning = preprocess_spectrum_with_im(
                    tf.constant(mz), tf.constant(intensity),
                    tf.constant(precursor_mz), tf.constant(precursor_charge),
                    tf.constant(ion_mobility), config
                )
                
                yield spectrum.numpy(), conditioning.numpy()
    
    output_signature = (
        tf.TensorSpec(shape=(config.N_BINS,), dtype=tf.float32),
        tf.TensorSpec(shape=(config.MAX_CHARGE + 2,), dtype=tf.float32),
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_scoring_dataset(file_list: List[str], batch_size: int, config: Config):
    """Create dataset for scoring all spectra."""
    
    def generator():
        for filepath in file_list:
            df = pd.read_parquet(filepath)
            
            for _, row in df.iterrows():
                mz = np.array(row['mz_array'], dtype=np.float32)
                intensity = np.array(row['intensity_array'], dtype=np.float32)
                precursor_mz = float(row['precursor_mz'])
                precursor_charge = float(row['precursor_charge'])
                ion_mobility = float(row['IM'])
                label = int(row['Outcome'])
                
                if np.isnan(ion_mobility) or ion_mobility <= 0:
                    continue
                
                spectrum, conditioning = preprocess_spectrum_with_im(
                    tf.constant(mz), tf.constant(intensity),
                    tf.constant(precursor_mz), tf.constant(precursor_charge),
                    tf.constant(ion_mobility), config
                )
                
                yield spectrum.numpy(), conditioning.numpy(), label
    
    output_signature = (
        tf.TensorSpec(shape=(config.N_BINS,), dtype=tf.float32),
        tf.TensorSpec(shape=(config.MAX_CHARGE + 2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ================================
# Learning Rate Schedule
# ================================

class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup + cosine decay."""
    
    def __init__(self, target_lr, warmup_steps, total_steps, min_lr_ratio=0.01):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = target_lr * min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        warmup_lr = self.target_lr * (step / tf.math.maximum(warmup_steps, 1.0))
        
        decay_steps = total_steps - warmup_steps
        decay_step = step - warmup_steps
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * decay_step / decay_steps))
        decay_lr = self.min_lr + (self.target_lr - self.min_lr) * cosine_decay
        
        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decay_lr)


# ================================
# Training Functions
# ================================
def train_conditional_autoencoder(parquet_dir: str, output_dir: str, seed: int = 42):
    """Train conditional autoencoder with precursor m/z, charge, ion mobility."""
    configure_environment(seed)
    config = Config()
    
    print("=" * 60)
    print("Conditional Autoencoder Training")
    print("=" * 60)
    print(f"Latent dimension: {config.LATENT_DIM}")
    print(f"Conditioning: precursor_mz, charge (one-hot), ion_mobility")
    print(f"Training on IDENTIFIED spectra only")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find files
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Split
    random.seed(seed)
    random.shuffle(parquet_files)
    n_val = max(2, len(parquet_files) // 10)
    train_files = parquet_files[:-n_val]
    val_files = parquet_files[-n_val:]
    
    # Create datasets (use the diffusion dataset which includes conditioning)
    train_ds = create_dataset(train_files, config.AE_BATCH_SIZE, config, identified_only=True)
    val_ds = create_dataset(val_files, config.AE_BATCH_SIZE, config, identified_only=True)
    
    # Conditioning dimension: one-hot charge (6) + precursor_mz (1) + ion_mobility (1) = 8
    cond_dim = config.MAX_CHARGE + 2
    
    # Create model
    autoencoder = ConditionalSpectrumAutoencoder(
        n_bins=config.N_BINS,
        patch_size=config.AE_PATCH_SIZE,
        embed_dim=config.AE_EMBED_DIM,
        num_heads=config.AE_NUM_HEADS,
        num_layers=config.AE_NUM_LAYERS,
        ff_dim=config.AE_FF_DIM,
        latent_dim=config.LATENT_DIM,
        cond_dim=cond_dim,
        dropout=config.AE_DROPOUT,
        use_kl=False,
    )
    
    # Compile
    steps_per_epoch = 1000
    total_steps = steps_per_epoch * config.AE_EPOCHS
    warmup_steps = steps_per_epoch * config.AE_WARMUP_EPOCHS
    
    lr_schedule = WarmUpCosineDecay(config.AE_INITIAL_LR, warmup_steps, total_steps)
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=config.WEIGHT_DECAY)
    autoencoder.compile(optimizer=optimizer)
    
    # Build
    dummy_spectrum = tf.zeros((2, config.N_BINS))
    dummy_cond = tf.zeros((2, cond_dim))
    _ = autoencoder((dummy_spectrum, dummy_cond), training=False)
    print(f"Conditional Autoencoder parameters: {autoencoder.count_params():,}")
    
    # Train
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_conditional_ae.weights.h5'),
            monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(output_dir, 'conditional_ae_training_log.csv')),
    ]
    
    history = autoencoder.fit(
        train_ds, validation_data=val_ds, epochs=config.AE_EPOCHS,
        callbacks=callbacks, verbose=2
    )
    
    # Save config
    import json
    ae_config = {
        'n_bins': config.N_BINS,
        'patch_size': config.AE_PATCH_SIZE,
        'embed_dim': config.AE_EMBED_DIM,
        'num_heads': config.AE_NUM_HEADS,
        'num_layers': config.AE_NUM_LAYERS,
        'ff_dim': config.AE_FF_DIM,
        'latent_dim': config.LATENT_DIM,
        'cond_dim': cond_dim,
        'dropout': config.AE_DROPOUT,
        'use_kl': False,
        'conditional': True,  # Flag to identify this as conditional
    }
    with open(os.path.join(output_dir, 'ae_config.json'), 'w') as f:
        json.dump(ae_config, f, indent=2)
    
    print(f"\nConditional Autoencoder saved to {output_dir}")
    return autoencoder, history

# ================================
# Diagnostic Functions
# ================================

def compute_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics for a batch.
    
    Args:
        original: [batch, n_bins] original spectra
        reconstructed: [batch, n_bins] reconstructed spectra
    
    Returns:
        Dictionary of metrics
    """
    # MSE
    mse = np.mean((original - reconstructed) ** 2, axis=1)
    
    # Cosine similarity
    orig_norm = original / (np.linalg.norm(original, axis=1, keepdims=True) + 1e-8)
    recon_norm = reconstructed / (np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.sum(orig_norm * recon_norm, axis=1)
    
    # Spectral angle (in radians)
    spectral_angle = np.arccos(np.clip(cosine_sim, -1, 1))
    
    # Peak preservation: what fraction of top-k peaks are in reconstruction's top-k?
    k = 10
    peak_preservation = []
    for orig, recon in zip(original, reconstructed):
        orig_top_k = set(np.argsort(orig)[-k:])
        recon_top_k = set(np.argsort(recon)[-k:])
        overlap = len(orig_top_k & recon_top_k) / k
        peak_preservation.append(overlap)
    peak_preservation = np.array(peak_preservation)
    
    return {
        'mse': mse,
        'cosine_similarity': cosine_sim,
        'spectral_angle_rad': spectral_angle,
        'top10_peak_preservation': peak_preservation,
    }


# ================================
# Clustering Analysis
# ================================
def encode_spectra_batched(spectra: np.ndarray, 
                           autoencoder, 
                           batch_size: int = 256,
                           latent_dim: int = 256) -> np.ndarray:
    """Encode spectra to latent space in batches."""
    n_spectra = len(spectra)
    latents = np.zeros((n_spectra, latent_dim), dtype=np.float32)
    
    print(f"Encoding {n_spectra:,} spectra to {latent_dim}-dim latent space...")
    start_time = time.time()
    
    for i in range(0, n_spectra, batch_size):
        end = min(i + batch_size, n_spectra)
        batch = spectra[i:end]
        
        # Encode
        z = autoencoder.encode(batch, training=False)
        if isinstance(z, tuple):
            z = z[0]  # VAE returns (z, mu, logvar)
        
        latents[i:end] = z.numpy()
        
        if (i // batch_size + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = end / elapsed
            eta = (n_spectra - end) / rate
            print(f"  Encoded {end:,}/{n_spectra:,} ({rate:.0f}/sec, ETA: {eta/60:.1f} min)")
    
    print(f"  Done in {(time.time() - start_time)/60:.1f} minutes")
    return latents


def cluster_latents(latents: np.ndarray,
                    method: str = 'hdbscan',
                    min_cluster_size: int = 5,
                    max_samples_for_full: int = 500000,
                    random_state: int = 42) -> Tuple[np.ndarray, dict]:
    """
    Cluster latent vectors using HDBSCAN or KMeans.
    
    For large datasets, uses a subsample+assign strategy.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    
    n_samples, n_dims = latents.shape
    print(f"\nClustering {n_samples:,} latents ({n_dims} dims)...")
    
    info = {'method': method, 'n_samples': n_samples}
    
    # Dimensionality reduction for efficiency
    if n_dims > 100:
        print(f"  Reducing dimensionality: {n_dims} -> 100 with PCA")
        pca = PCA(n_components=100, random_state=random_state)
        latents_reduced = pca.fit_transform(latents)
        info['pca_variance_explained'] = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA variance explained: {info['pca_variance_explained']:.3f}")
    else:
        latents_reduced = latents
    
    if method == 'hdbscan':
        try:
            import hdbscan
        except ImportError:
            print("  HDBSCAN not installed. Install with: pip install hdbscan")
            print("  Falling back to KMeans...")
            method = 'kmeans'
    
    if method == 'hdbscan':
        import hdbscan
        # Full HDBSCAN
        print(f"  Running HDBSCAN (min_cluster_size={min_cluster_size})...")
        start = time.time()
            
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=3,
            metric='euclidean',
            cluster_selection_method='eom',
            core_dist_n_jobs=-1
        )
        labels = clusterer.fit_predict(latents_reduced)
            
        info['time'] = time.time() - start
        info['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        info['n_noise'] = int((labels == -1).sum())
            
    
    elif method == 'kmeans':
        from sklearn.cluster import MiniBatchKMeans
        
        n_clusters = min(10000, n_samples // 10)
        print(f"  Running MiniBatchKMeans (k={n_clusters})...")
        start = time.time()
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            random_state=random_state,
            n_init=3
        )
        labels = kmeans.fit_predict(latents_reduced)
        
        info['time'] = time.time() - start
        info['n_clusters'] = len(set(labels))
        info['n_noise'] = 0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"  Found {info['n_clusters']:,} clusters, {info.get('n_noise', 0):,} noise points")
    
    return labels, info


def score_clusters(latents: np.ndarray,
                   labels: np.ndarray,
                   metadata: pd.DataFrame,
                   spectra: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Score each cluster by quality metrics.
    """
    print("\nScoring clusters...")
    
    unique_labels = sorted([l for l in set(labels) if l >= 0])
    print(f"  Scoring {len(unique_labels):,} clusters...")
    
    cluster_stats = []
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        n_members = mask.sum()
        
        if n_members < 2:
            continue
        
        cluster_latents = latents[mask]
        cluster_meta = metadata.iloc[np.where(mask)[0]]
        
        stats = {
            'cluster_id': label,
            'size': int(n_members),
        }
        
        # Ion mobility coherence
        if 'ion_mobility' in cluster_meta.columns:
            im_values = cluster_meta['ion_mobility'].values.astype(float)
            im_values = im_values[~np.isnan(im_values)]
            if len(im_values) > 0:
                stats['im_mean'] = float(np.mean(im_values))
                stats['im_std'] = float(np.std(im_values))
                stats['im_cv'] = stats['im_std'] / (stats['im_mean'] + 1e-6)
        
        # Precursor m/z coherence
        if 'precursor_mz' in cluster_meta.columns:
            mz_values = cluster_meta['precursor_mz'].values.astype(float)
            mz_values = mz_values[~np.isnan(mz_values)]
            if len(mz_values) > 0:
                stats['mz_mean'] = float(np.mean(mz_values))
                stats['mz_std'] = float(np.std(mz_values))
                stats['mz_range'] = float(np.max(mz_values) - np.min(mz_values))
        
        # Charge distribution
        if 'charge' in cluster_meta.columns:
            charges = cluster_meta['charge'].values
            stats['charge_mode'] = int(pd.Series(charges).mode().iloc[0])
            stats['charge_uniform'] = float((charges == stats['charge_mode']).mean())
        
        # File diversity
        if 'file' in cluster_meta.columns:
            stats['n_files'] = int(cluster_meta['file'].nunique())
        
        # Latent space coherence (cosine similarity)
        sample_size = min(n_members, 500)
        if n_members > sample_size:
            sample_idx = np.random.choice(n_members, size=sample_size, replace=False)
            sample_latents = cluster_latents[sample_idx]
        else:
            sample_latents = cluster_latents
        
        norms = np.linalg.norm(sample_latents, axis=1, keepdims=True)
        normalized = sample_latents / (norms + 1e-8)
        cos_sim = np.dot(normalized, normalized.T)
        triu_idx = np.triu_indices(len(sample_latents), k=1)
        stats['latent_coherence'] = float(cos_sim[triu_idx].mean())
        
        # Spectral coherence
        if spectra is not None and n_members <= 500:
            cluster_spectra = spectra[mask]
            spec_norms = np.linalg.norm(cluster_spectra, axis=1, keepdims=True)
            spec_normalized = cluster_spectra / (spec_norms + 1e-8)
            spec_cos_sim = np.dot(spec_normalized, spec_normalized.T)
            triu_idx = np.triu_indices(n_members, k=1)
            stats['spectral_coherence'] = float(spec_cos_sim[triu_idx].mean())
        
        cluster_stats.append(stats)
        
        if (i + 1) % 1000 == 0:
            print(f"    Scored {i+1:,}/{len(unique_labels):,} clusters")
    
    df = pd.DataFrame(cluster_stats)
    
    # Compute composite quality score
    if len(df) > 0:
        # Normalize components to [0, 1]
        df['quality_score'] = 0.0
        
        # Latent coherence (higher = better)
        if 'latent_coherence' in df.columns:
            df['quality_score'] += 0.3 * df['latent_coherence'].fillna(0)
        
        # Ion mobility coherence (lower CV = better)
        if 'im_cv' in df.columns:
            df['quality_score'] += 0.3 * (1 - df['im_cv'].fillna(1).clip(0, 1))
        
        # Charge uniformity (higher = better)
        if 'charge_uniform' in df.columns:
            df['quality_score'] += 0.2 * df['charge_uniform'].fillna(0)
        
        # Size bonus (log-scaled)
        df['quality_score'] += 0.1 * np.log1p(df['size']) / np.log1p(df['size'].max())
        
        # Multi-file bonus
        if 'n_files' in df.columns:
            df['quality_score'] += 0.1 * np.log1p(df['n_files']) / np.log1p(df['n_files'].max())
        
        df = df.sort_values('quality_score', ascending=False)
    
    print(f"  Scored {len(df):,} clusters")
    
    return df


def plot_cluster_analysis(cluster_df: pd.DataFrame,
                          latents: np.ndarray,
                          labels: np.ndarray,
                          output_dir: str,
                          max_clusters_to_show: int = 20):
    """Generate cluster analysis plots."""
    
    # 1. Cluster statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sizes = cluster_df['size'].values
    axes[0].hist(sizes, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Cluster Size')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Cluster Size Distribution')
    axes[0].set_yscale('log')
    
    axes[1].hist(cluster_df['quality_score'].values, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Quality Score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Cluster Quality Distribution')
    
    if 'im_std' in cluster_df.columns:
        scatter = axes[2].scatter(cluster_df['size'], cluster_df['im_std'], 
                                  c=cluster_df['quality_score'], cmap='viridis', 
                                  alpha=0.5, s=10)
        axes[2].set_xlabel('Cluster Size')
        axes[2].set_ylabel('Ion Mobility Std Dev')
        axes[2].set_title('Size vs IM Coherence')
        axes[2].set_xscale('log')
        plt.colorbar(scatter, ax=axes[2], label='Quality Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_statistics.png'), dpi=150)
    plt.close()
    
    # 2. UMAP visualization
    try:
        import umap
        
        print("  Computing UMAP for visualization...")
        
        n_sample = min(50000, len(latents))
        sample_idx = np.random.choice(len(latents), size=n_sample, replace=False)
        sample_latents = latents[sample_idx]
        sample_labels = labels[sample_idx]
        
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(sample_latents)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot noise
        noise_mask = sample_labels == -1
        ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                  c='lightgray', s=1, alpha=0.3, label='Noise/Unclustered')
        
        # Plot clustered points (color by cluster)
        clustered_mask = sample_labels >= 0
        scatter = ax.scatter(embedding[clustered_mask, 0], embedding[clustered_mask, 1],
                            c=sample_labels[clustered_mask], cmap='tab20', 
                            s=2, alpha=0.5)
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Latent Space Clustering ({len(cluster_df):,} clusters)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_umap.png'), dpi=150)
        plt.close()
        
    except ImportError:
        print("  UMAP not installed, skipping UMAP visualization")
    
    # 3. Top clusters detail
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    top_clusters = cluster_df.head(20)
    
    # Quality vs size
    axes[0, 0].scatter(top_clusters['size'], top_clusters['quality_score'], 
                       c='steelblue', s=50, alpha=0.7)
    axes[0, 0].set_xlabel('Cluster Size')
    axes[0, 0].set_ylabel('Quality Score')
    axes[0, 0].set_title('Top 20 Clusters: Size vs Quality')
    
    # IM std distribution
    if 'im_std' in top_clusters.columns:
        axes[0, 1].barh(range(len(top_clusters)), top_clusters['im_std'].values)
        axes[0, 1].set_yticks(range(len(top_clusters)))
        axes[0, 1].set_yticklabels([f"C{c}" for c in top_clusters['cluster_id'].values])
        axes[0, 1].set_xlabel('Ion Mobility Std Dev')
        axes[0, 1].set_title('Top 20: IM Coherence')
        axes[0, 1].invert_yaxis()
    
    # Latent coherence distribution
    if 'latent_coherence' in top_clusters.columns:
        axes[1, 0].barh(range(len(top_clusters)), top_clusters['latent_coherence'].values)
        axes[1, 0].set_yticks(range(len(top_clusters)))
        axes[1, 0].set_yticklabels([f"C{c}" for c in top_clusters['cluster_id'].values])
        axes[1, 0].set_xlabel('Latent Coherence (Cosine Sim)')
        axes[1, 0].set_title('Top 20: Latent Coherence')
        axes[1, 0].invert_yaxis()
    
    # Size distribution
    axes[1, 1].barh(range(len(top_clusters)), top_clusters['size'].values)
    axes[1, 1].set_yticks(range(len(top_clusters)))
    axes[1, 1].set_yticklabels([f"C{c}" for c in top_clusters['cluster_id'].values])
    axes[1, 1].set_xlabel('Cluster Size')
    axes[1, 1].set_title('Top 20: Cluster Size')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_clusters_detail.png'), dpi=150)
    plt.close()
    
    print(f"  Plots saved to {output_dir}")


def visualize_top_clusters(cluster_df: pd.DataFrame,
                           spectra: np.ndarray,
                           labels: np.ndarray,
                           metadata: pd.DataFrame,
                           output_dir: str,
                           n_clusters: int = 20,
                           n_spectra_per_cluster: int = 5,
                           config = None):
    """
    Visualize representative spectra from top clusters.
    
    For each top cluster:
    - Plot overlaid spectra showing consistency
    - Plot consensus spectrum
    - Show metadata summary
    """
    if config is None:
        config = Config()
    
    print(f"\nVisualizing top {n_clusters} clusters...")
    
    viz_dir = os.path.join(output_dir, 'cluster_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get top clusters
    top_clusters = cluster_df.head(n_clusters)
    
    # Create m/z axis for plotting
    mz_axis = np.linspace(config.MZ_MIN, config.MZ_MAX, config.N_BINS)
    
    # Summary file
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("TOP CLUSTER SUMMARY")
    summary_lines.append("=" * 80)
    
    for i, (_, row) in enumerate(top_clusters.iterrows()):
        cluster_id = int(row['cluster_id'])
        mask = labels == cluster_id
        cluster_spectra = spectra[mask]
        cluster_meta = metadata.iloc[np.where(mask)[0]]
        
        n_members = len(cluster_spectra)
        
        # Select representative spectra (random sample)
        if n_members > n_spectra_per_cluster:
            sample_idx = np.random.choice(n_members, size=n_spectra_per_cluster, replace=False)
        else:
            sample_idx = np.arange(n_members)
        
        sample_spectra = cluster_spectra[sample_idx]
        
        # Compute consensus spectrum (mean)
        consensus = cluster_spectra.mean(axis=0)
        consensus = consensus / (consensus.max() + 1e-8)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Overlaid spectra
        for j, spec in enumerate(sample_spectra):
            axes[0].plot(mz_axis, spec, alpha=0.5, linewidth=0.5)
        axes[0].set_xlabel('m/z')
        axes[0].set_ylabel('Relative Intensity')
        axes[0].set_title(f'Cluster {cluster_id}: Overlaid Spectra (n={n_members})')
        axes[0].set_xlim(config.MZ_MIN, config.MZ_MAX)
        
        # 2. Consensus spectrum (stem plot for clarity)
        # Find peaks above threshold
        peak_mask = consensus > 0.05
        peak_mz = mz_axis[peak_mask]
        peak_int = consensus[peak_mask]
        
        axes[1].stem(peak_mz, peak_int, linefmt='b-', markerfmt=' ', basefmt=' ')
        axes[1].set_xlabel('m/z')
        axes[1].set_ylabel('Relative Intensity')
        axes[1].set_title(f'Cluster {cluster_id}: Consensus Spectrum')
        axes[1].set_xlim(config.MZ_MIN, config.MZ_MAX)
        axes[1].set_ylim(0, 1.1)
        
        # 3. Metadata summary as text
        axes[2].axis('off')
        
        # Compute metadata stats
        mz_mean = cluster_meta['precursor_mz'].mean()
        mz_std = cluster_meta['precursor_mz'].std()
        charge_mode = int(cluster_meta['charge'].mode().iloc[0])
        charge_pct = (cluster_meta['charge'] == charge_mode).mean() * 100
        im_mean = cluster_meta['ion_mobility'].mean()
        im_std = cluster_meta['ion_mobility'].std()
        n_files = cluster_meta['file'].nunique()
        
        # Count significant peaks in consensus
        n_peaks = (consensus > 0.05).sum()
        
        info_text = f"""
CLUSTER {cluster_id} SUMMARY
{'='*40}

Size:               {n_members} spectra
Files:              {n_files}

Precursor m/z:      {mz_mean:.2f} ± {mz_std:.2f}
Charge:             {charge_mode}+ ({charge_pct:.0f}% uniform)
Ion mobility (1/K₀): {im_mean:.4f} ± {im_std:.4f} Vs/cm²

Quality Score:      {row['quality_score']:.4f}
Latent Coherence:   {row.get('latent_coherence', 0):.4f}

Consensus Peaks:    {n_peaks} (>5% relative)

Top 10 Peak m/z values:
{', '.join([f'{m:.1f}' for m in peak_mz[np.argsort(peak_int)[-10:]]])}
"""
        
        axes[2].text(0.1, 0.9, info_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'cluster_{cluster_id:04d}.png'), dpi=150)
        plt.close()
        
        # Add to summary
        summary_lines.append(f"\nCluster {cluster_id}:")
        summary_lines.append(f"  Size: {n_members}, Files: {n_files}")
        summary_lines.append(f"  Precursor m/z: {mz_mean:.2f} ± {mz_std:.2f}")
        summary_lines.append(f"  Charge: {charge_mode}+, IM: {im_mean:.4f} ± {im_std:.4f}")
        summary_lines.append(f"  Quality: {row['quality_score']:.4f}")
    
    # Save summary text
    with open(os.path.join(viz_dir, 'cluster_summary.txt'), 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Create overview grid of consensus spectra
    n_cols = 5
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(top_clusters.iterrows()):
        if i >= len(axes):
            break
            
        cluster_id = int(row['cluster_id'])
        mask = labels == cluster_id
        cluster_spectra = spectra[mask]
        
        consensus = cluster_spectra.mean(axis=0)
        consensus = consensus / (consensus.max() + 1e-8)
        
        axes[i].fill_between(mz_axis, 0, consensus, alpha=0.7)
        axes[i].set_title(f"C{cluster_id} (n={mask.sum()})", fontsize=10)
        axes[i].set_xlim(config.MZ_MIN, config.MZ_MAX)
        axes[i].set_ylim(0, 1)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Hide empty subplots
    for i in range(len(top_clusters), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Top Cluster Consensus Spectra', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_clusters_consensus.png'), dpi=150)
    plt.close()
    
    print(f"  Saved {n_clusters} cluster visualizations to {viz_dir}/")
    print(f"  Saved consensus overview to {output_dir}/top_clusters_consensus.png")


def cluster_dark_proteome(parquet_dir: str, 
                          autoencoder_dir: str, 
                          output_dir: str,
                          seed: int = 42,
                          max_spectra: Optional[int] = None,
                          min_cluster_size: int = 5,
                          method: str = 'hdbscan',
                          force_reencode: bool = False,
                          unidentified_only: bool = True):
    """
    Cluster spectra to find candidate novel peptides.
    
    Args:
        unidentified_only: If True, only cluster unidentified spectra (Outcome=0).
                          If False, cluster ALL spectra (for validation with known peptides).
    
    Memory-efficient version that streams data and saves intermediates to disk.
    """
    configure_environment(seed)
    config = Config()
    
    print("=" * 60)
    if unidentified_only:
        print("Dark Proteome Clustering Analysis")
        print("  Mode: Unidentified spectra only (discovery)")
    else:
        print("Full Spectrum Clustering Analysis")
        print("  Mode: ALL spectra (validation)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load autoencoder
    print("\nLoading autoencoder...")
    with open(os.path.join(autoencoder_dir, 'ae_config.json'), 'r') as f:
        ae_config = json.load(f)
    
    latent_dim = ae_config.get('latent_dim', config.LATENT_DIM)
    is_conditional = ae_config.get('conditional', False)
    cond_dim = ae_config.get('cond_dim', config.MAX_CHARGE + 2)
    
    if is_conditional:
        print("Loading CONDITIONAL autoencoder...")
        autoencoder = ConditionalSpectrumAutoencoder(**{k: v for k, v in ae_config.items() if k != 'conditional'})
        dummy_spectrum = tf.zeros((2, config.N_BINS))
        dummy_cond = tf.zeros((2, cond_dim))
        _ = autoencoder((dummy_spectrum, dummy_cond), training=False)
        autoencoder.load_weights(os.path.join(autoencoder_dir, 'best_conditional_ae.weights.h5'))
    else:
        print("Conditional autoencoder not found, use correct directory please.")
    print("Autoencoder loaded.")
    
    # Find parquet files
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} parquet files")
    
    # ========================================
    # PHASE 1: Stream through data, encode latents, save to disk
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 1: Encoding spectra (streaming)")
    print("=" * 60)
    
    latents_file = os.path.join(output_dir, 'latents.npy')
    metadata_file = os.path.join(output_dir, 'metadata.parquet')
    
    # Delete old files if force re-encode
    if force_reencode:
        print("Force re-encode requested. Deleting cached files...")
        for f in [latents_file, metadata_file]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Deleted {f}")
    
    # Check if we can resume from existing latents
    if os.path.exists(latents_file) and os.path.exists(metadata_file) and not force_reencode:
        print("Found existing latents and metadata. Loading...")
        print("  (Use --force to re-encode from scratch)")
        latents = np.load(latents_file)
        metadata = pd.read_parquet(metadata_file)
        total_spectra = len(latents)
        print(f"Loaded {total_spectra:,} pre-computed latents")
    else:
        # First pass: count spectra and detect columns
        print("Analyzing parquet structure...")
        sample_df = pd.read_parquet(parquet_files[0])
        print(f"  Columns: {list(sample_df.columns)}")
        
        # Check for key columns
        if 'IM' in sample_df.columns:
            im_vals = sample_df['IM'].dropna()
            if len(im_vals) > 0:
                print(f"  Ion mobility (IM): {im_vals.min():.4f} - {im_vals.max():.4f}")
        else:
            print("  WARNING: No 'IM' column found!")
        
        if 'Outcome' in sample_df.columns:
            print(f"  Outcome distribution: {sample_df['Outcome'].value_counts().to_dict()}")
        
        del sample_df
        gc.collect()
        
        print("\nCounting spectra...")
        total_spectra = 0
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf, columns=['Outcome'])
                if 'Outcome' in df.columns and unidentified_only:
                    total_spectra += (df['Outcome'] == 0).sum()
                else:
                    total_spectra += len(df)
            except:
                pass
        
        if max_spectra:
            total_spectra = min(total_spectra, max_spectra)
        
        if unidentified_only:
            print(f"Total unidentified spectra: {total_spectra:,}")
        else:
            print(f"Total spectra (all): {total_spectra:,}")
        
        # Pre-allocate memory-mapped array for latents
        print(f"Creating memory-mapped latent array ({total_spectra:,} x {latent_dim})...")
        latents_mmap = np.lib.format.open_memmap(
            latents_file, mode='w+', dtype=np.float32, shape=(total_spectra, latent_dim)
        )
        
        # Stream through files, encode, save
        all_metadata = []
        current_idx = 0
        batch_spectra = []
        batch_conditioning = []  # For conditional AE
        batch_size = 256
        
        start_time = time.time()
        
        for file_idx, pf in enumerate(parquet_files):
            if max_spectra and current_idx >= max_spectra:
                break
            
            try:
                df = pd.read_parquet(pf)
            except Exception as e:
                print(f"  Error reading {pf}: {e}")
                continue
            
            # Filter to unidentified only (if requested)
            if unidentified_only and 'Outcome' in df.columns:
                df = df[df['Outcome'] == 0]
            
            if len(df) == 0:
                continue
            
            # Limit if needed
            if max_spectra and (current_idx + len(df)) > max_spectra:
                df = df.iloc[:max_spectra - current_idx]
            
            # Process each spectrum
            for idx, row in df.iterrows():
                mz = np.array(row['mz_array'], dtype=np.float32)
                intensity = np.array(row['intensity_array'], dtype=np.float32)
                
                binned = bin_spectrum_numpy(mz, intensity, config)
                batch_spectra.append(binned)
                
                # Collect metadata
                precursor_mz = row.get('precursor_mz', row.get('PrecursorMz', 0))
                charge = row.get('precursor_charge', row.get('PrecursorCharge', row.get('charge', 2)))
                ion_mobility = row.get('IM', row.get('IonMobility', row.get('ion_mobility', 0.0)))
                
                meta = {
                    'precursor_mz': precursor_mz,
                    'charge': charge,
                    'ion_mobility': ion_mobility,
                    'file': os.path.basename(pf),
                }
                if 'scanID' in row:
                    meta['scan'] = row['scanID']
                elif 'ScanNumber' in row:
                    meta['scan'] = row['ScanNumber']
                
                # Add Outcome and peptide for validation (when clustering all spectra)
                if 'Outcome' in row:
                    meta['Outcome'] = row['Outcome']
                if 'peptide' in row:
                    meta['peptide'] = row['peptide']
                if 'modified_peptide' in row:
                    meta['modified_peptide'] = row['modified_peptide']
                
                all_metadata.append(meta)
                
                # Collect conditioning for conditional AE
                if is_conditional:
                    # Build conditioning vector: one-hot charge + normalized mz + normalized IM
                    charge_int = int(charge) if not np.isnan(charge) else 2
                    charge_int = max(1, min(charge_int, config.MAX_CHARGE))
                    charge_onehot = np.zeros(config.MAX_CHARGE, dtype=np.float32)
                    charge_onehot[charge_int - 1] = 1.0
                    
                    mz_norm = float(precursor_mz) / config.PRECURSOR_MZ_MAX if precursor_mz else 0.5
                    im_norm = (float(ion_mobility) - config.IM_MIN) / (config.IM_MAX - config.IM_MIN) if ion_mobility else 0.5
                    im_norm = max(0.0, min(1.0, im_norm))
                    
                    cond_vec = np.concatenate([charge_onehot, [mz_norm, im_norm]])
                    batch_conditioning.append(cond_vec)
                
                # Encode batch when full
                if len(batch_spectra) >= batch_size:
                    batch_array = np.array(batch_spectra, dtype=np.float32)
                    
                    if is_conditional:
                        cond_array = np.array(batch_conditioning, dtype=np.float32)
                        z = autoencoder.encode(batch_array, cond_array, training=False)
                    else:
                        z = autoencoder.encode(batch_array, training=False)
                    
                    if isinstance(z, tuple):
                        z = z[0]
                    
                    batch_len = len(batch_spectra)
                    latents_mmap[current_idx:current_idx + batch_len] = z.numpy()
                    current_idx += batch_len
                    batch_spectra = []
                    batch_conditioning = []
                    
                    # Flush periodically
                    if current_idx % 10000 == 0:
                        latents_mmap.flush()
            
            # Progress update
            if (file_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = current_idx / elapsed
                eta = (total_spectra - current_idx) / rate if rate > 0 else 0
                print(f"  Files: {file_idx+1}/{len(parquet_files)}, "
                      f"Spectra: {current_idx:,}/{total_spectra:,} "
                      f"({rate:.0f}/sec, ETA: {eta/60:.1f} min)")
            
            del df
            gc.collect()
        
        # Encode remaining batch
        if batch_spectra:
            batch_array = np.array(batch_spectra, dtype=np.float32)
            
            if is_conditional:
                cond_array = np.array(batch_conditioning, dtype=np.float32)
                z = autoencoder.encode(batch_array, cond_array, training=False)
            else:
                z = autoencoder.encode(batch_array, training=False)
            
            if isinstance(z, tuple):
                z = z[0]
            batch_len = len(batch_spectra)
            latents_mmap[current_idx:current_idx + batch_len] = z.numpy()
            current_idx += batch_len
        
        # Trim if we got fewer spectra than expected
        if current_idx < total_spectra:
            print(f"  Trimming array: {total_spectra} -> {current_idx}")
            latents_mmap.flush()
            del latents_mmap
            
            # Reload and trim
            latents = np.load(latents_file)[:current_idx].copy()
            np.save(latents_file, latents)
            total_spectra = current_idx
        else:
            latents_mmap.flush()
            del latents_mmap
            latents = np.load(latents_file)
        
        # Save metadata
        metadata = pd.DataFrame(all_metadata[:total_spectra])
        metadata.to_parquet(metadata_file)
        
        print(f"\nEncoding complete: {total_spectra:,} spectra in {(time.time()-start_time)/60:.1f} min")
        print(f"Saved latents to {latents_file}")
        print(f"Saved metadata to {metadata_file}")
        
        del all_metadata
        gc.collect()
    
    # ========================================
    # PHASE 2: Cluster latents
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 2: Clustering")
    print("=" * 60)
    
    labels, cluster_info = cluster_latents(
        latents,
        method=method,
        min_cluster_size=min_cluster_size,
        random_state=seed
    )
    
    np.save(os.path.join(output_dir, 'cluster_labels.npy'), labels)
    
    # ========================================
    # PHASE 3: Score clusters (memory-efficient)
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 3: Scoring clusters")
    print("=" * 60)
    
    cluster_df = score_clusters_lightweight(latents, labels, metadata)
    cluster_df.to_csv(os.path.join(output_dir, 'cluster_scores.csv'), index=False)
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    
    n_clustered = int((labels >= 0).sum())
    n_noise = int((labels == -1).sum())
    
    print(f"Total spectra: {len(latents):,}")
    print(f"Clustered: {n_clustered:,} ({100*n_clustered/len(latents):.1f}%)")
    print(f"Noise: {n_noise:,} ({100*n_noise/len(latents):.1f}%)")
    print(f"Clusters found: {cluster_info['n_clusters']:,}")
    
    high_quality = pd.DataFrame()
    if len(cluster_df) > 0:
        print(f"\nTop 10 clusters by quality score:")
        print(cluster_df.head(10).to_string())
        
        high_quality = cluster_df[
            (cluster_df['quality_score'] > 0.5) &
            (cluster_df['size'] >= 5)
        ]
        print(f"\nHigh-quality clusters (score > 0.5, size >= 5): {len(high_quality):,}")
        
        if 'im_std' in cluster_df.columns:
            coherent = cluster_df[cluster_df['im_std'] < 0.05]
            print(f"Clusters with coherent ion mobility (std < 0.05): {len(coherent):,}")
    
    # ========================================
    # Visualizations
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 4: Visualizations")
    print("=" * 60)
    
    plot_cluster_analysis(cluster_df, latents, labels, output_dir)
    
    # Save summary
    summary = {
        'n_spectra': len(latents),
        'n_clustered': n_clustered,
        'n_noise': n_noise,
        'n_clusters': cluster_info['n_clusters'],
        'method': method,
        'min_cluster_size': min_cluster_size,
        'high_quality_clusters': len(high_quality) if len(cluster_df) > 0 else 0,
    }
    
    with open(os.path.join(output_dir, 'clustering_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print("\nNote: Run 'visualize-clusters' command separately to generate detailed cluster plots")
    
    return cluster_df, labels, metadata


def score_clusters_lightweight(latents: np.ndarray,
                               labels: np.ndarray,
                               metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Score clusters without loading spectra (memory-efficient).
    """
    print("Scoring clusters (lightweight mode)...")
    
    unique_labels = sorted([l for l in set(labels) if l >= 0])
    print(f"  Scoring {len(unique_labels):,} clusters...")
    
    cluster_stats = []
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        n_members = mask.sum()
        
        if n_members < 2:
            continue
        
        cluster_latents = latents[mask]
        cluster_meta = metadata.iloc[np.where(mask)[0]]
        
        stats = {
            'cluster_id': label,
            'size': int(n_members),
        }
        
        # Ion mobility coherence
        if 'ion_mobility' in cluster_meta.columns:
            im_values = cluster_meta['ion_mobility'].values.astype(float)
            im_values = im_values[~np.isnan(im_values)]
            if len(im_values) > 0:
                stats['im_mean'] = float(np.mean(im_values))
                stats['im_std'] = float(np.std(im_values))
                stats['im_cv'] = stats['im_std'] / (stats['im_mean'] + 1e-6)
        
        # Precursor m/z coherence
        if 'precursor_mz' in cluster_meta.columns:
            mz_values = cluster_meta['precursor_mz'].values.astype(float)
            mz_values = mz_values[~np.isnan(mz_values)]
            if len(mz_values) > 0:
                stats['mz_mean'] = float(np.mean(mz_values))
                stats['mz_std'] = float(np.std(mz_values))
                stats['mz_range'] = float(np.max(mz_values) - np.min(mz_values))
        
        # Charge distribution
        if 'charge' in cluster_meta.columns:
            charges = cluster_meta['charge'].values
            stats['charge_mode'] = int(pd.Series(charges).mode().iloc[0])
            stats['charge_uniform'] = float((charges == stats['charge_mode']).mean())
        
        # File diversity
        if 'file' in cluster_meta.columns:
            stats['n_files'] = int(cluster_meta['file'].nunique())
        
        # Latent space coherence (sample for large clusters)
        sample_size = min(n_members, 500)
        if n_members > sample_size:
            sample_idx = np.random.choice(n_members, size=sample_size, replace=False)
            sample_latents = cluster_latents[sample_idx]
        else:
            sample_latents = cluster_latents
        
        norms = np.linalg.norm(sample_latents, axis=1, keepdims=True)
        normalized = sample_latents / (norms + 1e-8)
        cos_sim = np.dot(normalized, normalized.T)
        triu_idx = np.triu_indices(len(sample_latents), k=1)
        stats['latent_coherence'] = float(cos_sim[triu_idx].mean())
        
        cluster_stats.append(stats)
        
        if (i + 1) % 5000 == 0:
            print(f"    Scored {i+1:,}/{len(unique_labels):,} clusters")
    
    df = pd.DataFrame(cluster_stats)
    
    # Compute composite quality score
    if len(df) > 0:
        df['quality_score'] = 0.0
        
        if 'latent_coherence' in df.columns:
            df['quality_score'] += 0.3 * df['latent_coherence'].fillna(0)
        
        if 'im_cv' in df.columns:
            df['quality_score'] += 0.3 * (1 - df['im_cv'].fillna(1).clip(0, 1))
        
        if 'charge_uniform' in df.columns:
            df['quality_score'] += 0.2 * df['charge_uniform'].fillna(0)
        
        df['quality_score'] += 0.1 * np.log1p(df['size']) / np.log1p(df['size'].max())
        
        if 'n_files' in df.columns:
            df['quality_score'] += 0.1 * np.log1p(df['n_files']) / np.log1p(df['n_files'].max())
        
        df = df.sort_values('quality_score', ascending=False)
    
    print(f"  Scored {len(df):,} clusters")
    
    return df


def bin_spectrum_numpy(mz: np.ndarray, intensity: np.ndarray, config) -> np.ndarray:
    """Bin a single spectrum (numpy version matching TF preprocessing)."""
    # Filter to valid range
    mask = (mz >= config.MZ_MIN) & (mz < config.MZ_MAX) & (intensity > 0)
    mz = mz[mask]
    intensity = intensity[mask]
    
    if len(intensity) == 0:
        return np.zeros(config.N_BINS, dtype=np.float32)
    
    # Normalize
    intensity = intensity / intensity.max()
    
    # Relative intensity threshold
    mask = intensity >= config.RELATIVE_INTENSITY_THRESHOLD
    mz = mz[mask]
    intensity = intensity[mask]
    
    if len(intensity) == 0:
        return np.zeros(config.N_BINS, dtype=np.float32)
    
    # Top N peaks
    if hasattr(config, 'TOP_N_PEAKS') and config.TOP_N_PEAKS and len(intensity) > config.TOP_N_PEAKS:
        top_idx = np.argsort(intensity)[-config.TOP_N_PEAKS:]
        mz = mz[top_idx]
        intensity = intensity[top_idx]
    
    # Sqrt transform
    intensity = np.sqrt(intensity)
    
    # Bin
    bin_indices = ((mz - config.MZ_MIN) / config.BIN_SIZE).astype(int)
    bin_indices = np.clip(bin_indices, 0, config.N_BINS - 1)
    
    binned = np.zeros(config.N_BINS, dtype=np.float32)
    np.maximum.at(binned, bin_indices, intensity)
    
    # Final normalization
    if binned.max() > 0:
        binned = binned / binned.max()
    
    return binned


def visualize_clusters_from_disk(parquet_dir: str,
                                  cluster_dir: str,
                                  output_dir: str,
                                  n_clusters: int = 50,
                                  n_spectra_per_cluster: int = 5):
    """
    Generate detailed visualizations of top clusters.
    
    Loads spectra on-demand to avoid memory issues.
    """
    config = Config()
    
    print("=" * 60)
    print("Cluster Visualization")
    print("=" * 60)
    
    # Load clustering results
    print("\nLoading clustering results...")
    cluster_df = pd.read_csv(os.path.join(cluster_dir, 'cluster_scores.csv'))
    labels = np.load(os.path.join(cluster_dir, 'cluster_labels.npy'))
    metadata = pd.read_parquet(os.path.join(cluster_dir, 'metadata.parquet'))
    
    print(f"Loaded {len(labels):,} spectra, {len(cluster_df):,} clusters")
    
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'cluster_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get top clusters
    top_clusters = cluster_df.head(n_clusters)
    
    # Build file index for efficient spectrum retrieval
    print("Building file index...")
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    file_to_path = {os.path.basename(pf): pf for pf in parquet_files}
    
    # m/z axis for plotting
    mz_axis = np.linspace(config.MZ_MIN, config.MZ_MAX, config.N_BINS)
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("TOP CLUSTER SUMMARY")
    summary_lines.append("=" * 80)
    
    print(f"\nVisualizing top {n_clusters} clusters...")
    
    for i, (_, row) in enumerate(top_clusters.iterrows()):
        cluster_id = int(row['cluster_id'])
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_meta = metadata.iloc[cluster_indices]
        n_members = len(cluster_indices)
        
        # Select subset of spectra to load
        if n_members > n_spectra_per_cluster * 10:
            sample_indices = np.random.choice(len(cluster_indices), 
                                             size=n_spectra_per_cluster * 10, 
                                             replace=False)
            sample_meta = cluster_meta.iloc[sample_indices]
        else:
            sample_meta = cluster_meta
        
        # Load spectra from parquet files
        sample_spectra = []
        for _, spec_row in sample_meta.iterrows():
            file_path = file_to_path.get(spec_row['file'])
            if file_path is None:
                continue
            
            try:
                # Load file and find spectrum
                df = pd.read_parquet(file_path)
                
                # Filter to unidentified
                if 'Outcome' in df.columns:
                    df = df[df['Outcome'] == 0]
                
                # Find matching spectrum by precursor_mz (approximate)
                target_mz = spec_row['precursor_mz']
                matches = df[np.abs(df['precursor_mz'] - target_mz) < 0.01]
                
                if len(matches) > 0:
                    match_row = matches.iloc[0]
                    mz = np.array(match_row['mz_array'], dtype=np.float32)
                    intensity = np.array(match_row['intensity_array'], dtype=np.float32)
                    binned = bin_spectrum_numpy(mz, intensity, config)
                    sample_spectra.append(binned)
                
                del df
                
                if len(sample_spectra) >= n_spectra_per_cluster * 5:
                    break
                    
            except Exception as e:
                continue
        
        if len(sample_spectra) < 2:
            print(f"  Cluster {cluster_id}: Could not load enough spectra, skipping")
            continue
        
        sample_spectra = np.array(sample_spectra)
        
        # Compute consensus
        consensus = sample_spectra.mean(axis=0)
        consensus = consensus / (consensus.max() + 1e-8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Overlaid spectra
        for j, spec in enumerate(sample_spectra[:n_spectra_per_cluster]):
            axes[0].plot(mz_axis, spec, alpha=0.5, linewidth=0.5)
        axes[0].set_xlabel('m/z')
        axes[0].set_ylabel('Relative Intensity')
        axes[0].set_title(f'Cluster {cluster_id}: Overlaid Spectra (n={n_members})')
        axes[0].set_xlim(config.MZ_MIN, config.MZ_MAX)
        
        # 2. Consensus spectrum
        peak_mask = consensus > 0.05
        peak_mz = mz_axis[peak_mask]
        peak_int = consensus[peak_mask]
        
        axes[1].stem(peak_mz, peak_int, linefmt='b-', markerfmt=' ', basefmt=' ')
        axes[1].set_xlabel('m/z')
        axes[1].set_ylabel('Relative Intensity')
        axes[1].set_title(f'Cluster {cluster_id}: Consensus Spectrum')
        axes[1].set_xlim(config.MZ_MIN, config.MZ_MAX)
        axes[1].set_ylim(0, 1.1)
        
        # 3. Metadata summary
        axes[2].axis('off')
        
        mz_mean = cluster_meta['precursor_mz'].mean()
        mz_std = cluster_meta['precursor_mz'].std()
        charge_mode = int(cluster_meta['charge'].mode().iloc[0])
        charge_pct = (cluster_meta['charge'] == charge_mode).mean() * 100
        im_mean = cluster_meta['ion_mobility'].mean()
        im_std = cluster_meta['ion_mobility'].std()
        n_files = cluster_meta['file'].nunique()
        n_peaks = (consensus > 0.05).sum()
        
        info_text = f"""
CLUSTER {cluster_id} SUMMARY
{'='*40}

Size:               {n_members} spectra
Files:              {n_files}

Precursor m/z:      {mz_mean:.2f} ± {mz_std:.2f}
Charge:             {charge_mode}+ ({charge_pct:.0f}% uniform)
Ion mobility (1/K₀): {im_mean:.4f} ± {im_std:.4f} Vs/cm²

Quality Score:      {row['quality_score']:.4f}
Latent Coherence:   {row.get('latent_coherence', 0):.4f}

Consensus Peaks:    {n_peaks} (>5% relative)

Top 10 Peak m/z values:
{', '.join([f'{m:.1f}' for m in peak_mz[np.argsort(peak_int)[-10:]]]) if len(peak_mz) > 0 else 'N/A'}
"""
        
        axes[2].text(0.1, 0.9, info_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'cluster_{cluster_id:04d}.png'), dpi=150)
        plt.close()
        
        summary_lines.append(f"\nCluster {cluster_id}:")
        summary_lines.append(f"  Size: {n_members}, Files: {n_files}")
        summary_lines.append(f"  Precursor m/z: {mz_mean:.2f} ± {mz_std:.2f}")
        summary_lines.append(f"  Charge: {charge_mode}+, IM: {im_mean:.4f} ± {im_std:.4f}")
        summary_lines.append(f"  Quality: {row['quality_score']:.4f}")
        
        if (i + 1) % 10 == 0:
            print(f"  Visualized {i+1}/{n_clusters} clusters")
        
        gc.collect()
    
    # Save summary
    with open(os.path.join(viz_dir, 'cluster_summary.txt'), 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nSaved visualizations to {viz_dir}/")
    print(f"Saved summary to {viz_dir}/cluster_summary.txt")


# ================================
# Main
# ================================

def main():
    parser = ArgumentParser(description="SpecDiff Latent - Latent Diffusion for MS/MS")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train conditional autoencoder
    cond_ae_parser = subparsers.add_parser('train-conditional-ae', 
                                           help='Train conditional autoencoder with precursor/IM')
    cond_ae_parser.add_argument('-i', '--input', required=True, help='Parquet directory')
    cond_ae_parser.add_argument('-o', '--output', required=True, help='Output directory')
    cond_ae_parser.add_argument('--seed', type=int, default=42)
    
    # Cluster dark proteome
    cluster_parser = subparsers.add_parser('cluster', help='Cluster unidentified spectra')
    cluster_parser.add_argument('-i', '--input', required=True, help='Parquet directory')
    cluster_parser.add_argument('-a', '--autoencoder', required=True, help='Autoencoder directory')
    cluster_parser.add_argument('-o', '--output', required=True, help='Output directory')
    cluster_parser.add_argument('--seed', type=int, default=42)
    cluster_parser.add_argument('--max-spectra', type=int, default=None,
                               help='Max spectra to cluster (default: all)')
    cluster_parser.add_argument('--min-cluster-size', type=int, default=5,
                               help='Minimum cluster size for HDBSCAN (default: 5)')
    cluster_parser.add_argument('--method', choices=['hdbscan', 'kmeans'], default='hdbscan',
                               help='Clustering method (default: hdbscan)')
    cluster_parser.add_argument('--force', action='store_true',
                               help='Force re-encoding even if latents.npy exists')
    cluster_parser.add_argument('--all', action='store_true', dest='include_all',
                               help='Cluster ALL spectra (not just unidentified). Use for validation.')
    
    # Visualize clusters
    viz_parser = subparsers.add_parser('visualize-clusters', help='Visualize top clusters')
    viz_parser.add_argument('-i', '--input', required=True, help='Parquet directory (original data)')
    viz_parser.add_argument('-c', '--clusters', required=True, help='Clustering output directory')
    viz_parser.add_argument('-o', '--output', required=True, help='Output directory for visualizations')
    viz_parser.add_argument('--n-clusters', type=int, default=50, help='Number of top clusters to visualize')
    
    args = parser.parse_args()
    
    if args.command == 'train-conditional-ae':
        train_conditional_autoencoder(args.input, args.output, args.seed)
    elif args.command == 'cluster':
        cluster_dark_proteome(args.input, args.autoencoder, args.output,
                             args.seed, args.max_spectra, args.min_cluster_size, 
                             args.method, args.force, 
                             unidentified_only=not args.include_all)
    elif args.command == 'visualize-clusters':
        visualize_clusters_from_disk(args.input, args.clusters, args.output, args.n_clusters)


if __name__ == '__main__':
    main()