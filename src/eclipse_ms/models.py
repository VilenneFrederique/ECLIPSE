"""Conditional spectrum autoencoder models.

Ported verbatim from the training code. The encoder is the only part needed for
embedding/clustering; the decoder and the full autoencoder (with train/test
steps) are included so the same classes can load full weights and be retrained.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .layers import PatchEmbedding, TransformerBlock


@keras.utils.register_keras_serializable()
class ConditionalSpectrumEncoder(keras.Model):
    """Encode a binned spectrum + conditioning vector to a latent vector."""

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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_bins = n_bins
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.num_patches = n_bins // patch_size

        self.patch_embed = PatchEmbedding(embed_dim, patch_size)

        self.cond_proj = keras.Sequential(
            [
                layers.Dense(embed_dim, activation="gelu"),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(embed_dim),
            ],
            name="cond_projection",
        )

        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.num_patches + 2, embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

        self.final_norm = layers.LayerNormalization(epsilon=1e-6)

        self.to_latent = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="gelu"),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(latent_dim),
            ]
        )

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
            "n_bins": self.n_bins,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
        }


@keras.utils.register_keras_serializable()
class ConditionalSpectrumDecoder(keras.Model):
    """Two-head conditional decoder: latent + conditioning -> spectrum."""

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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_bins = n_bins
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.num_patches = n_bins // patch_size

        self.cond_proj = keras.Sequential(
            [
                layers.Dense(embed_dim, activation="gelu"),
                layers.Dense(embed_dim),
            ],
            name="cond_projection",
        )

        self.from_latent = keras.Sequential(
            [
                layers.Dense(embed_dim * 4, activation="gelu"),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(embed_dim * self.num_patches),
                layers.Reshape((self.num_patches, embed_dim)),
            ]
        )

        self.pos_embed = self.add_weight(
            name="dec_pos_embed",
            shape=(1, self.num_patches + 1, embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

        self.final_norm = layers.LayerNormalization(epsilon=1e-6)

        self.presence_head = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dense(patch_size, dtype="float32"),
            ],
            name="presence_head",
        )

        self.intensity_head = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dense(patch_size, dtype="float32"),
            ],
            name="intensity_head",
        )

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
            "n_bins": self.n_bins,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
        }


@keras.utils.register_keras_serializable()
class ConditionalSpectrumAutoencoder(keras.Model):
    """Conditional autoencoder: (spectrum, conditioning) -> latent -> spectrum."""

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
        **kwargs,
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

        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.cosine_sim_tracker = keras.metrics.Mean(name="cosine_sim")
        self.sparsity_tracker = keras.metrics.Mean(name="sparsity")
        self.presence_acc_tracker = keras.metrics.Mean(name="presence_acc")
        self.fp_rate_tracker = keras.metrics.Mean(name="fp_rate")

    def encode(self, x, cond, training=False):
        z = self.encoder((x, cond), training=training)

        if self.use_kl:
            mu = z[:, : self.latent_dim]
            logvar = z[:, self.latent_dim :]

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

    def _compute_losses(self, x, x_recon):
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

        false_positive_mask = 1 - peak_mask
        false_positive_penalty = tf.reduce_mean(presence_prob * false_positive_mask)

        recon_loss = (
            1.0 * presence_loss
            + 1.0 * intensity_loss
            + 0.5 * spectral_angle_loss
            + 0.5 * false_positive_penalty
        )
        return recon_loss, cos_sim, presence_prob, peak_mask

    def _update_trackers(self, recon_loss, kl_loss, total_loss, cos_sim, presence_prob, peak_mask):
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

    def _results(self):
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "cosine_sim": self.cosine_sim_tracker.result(),
            "sparsity": self.sparsity_tracker.result(),
            "presence_acc": self.presence_acc_tracker.result(),
            "fp_rate": self.fp_rate_tracker.result(),
        }

    def train_step(self, data):
        x, cond = data
        with tf.GradientTape() as tape:
            if self.use_kl:
                z, mu, logvar = self.encode(x, cond, training=True)
                x_recon = self.decode(z, cond, training=True)
                kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
            else:
                z = self.encode(x, cond, training=True)
                x_recon = self.decode(z, cond, training=True)
                kl_loss = 0.0

            recon_loss, cos_sim, presence_prob, peak_mask = self._compute_losses(x, x_recon)
            total_loss = recon_loss
            if self.use_kl:
                total_loss += self.kl_weight * kl_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self._update_trackers(recon_loss, kl_loss, total_loss, cos_sim, presence_prob, peak_mask)
        return self._results()

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

        recon_loss, cos_sim, presence_prob, peak_mask = self._compute_losses(x, x_recon)
        total_loss = recon_loss
        if self.use_kl:
            total_loss += self.kl_weight * kl_loss
        self._update_trackers(recon_loss, kl_loss, total_loss, cos_sim, presence_prob, peak_mask)
        return self._results()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.cosine_sim_tracker,
            self.sparsity_tracker,
            self.presence_acc_tracker,
            self.fp_rate_tracker,
        ]

    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
            "use_kl": self.use_kl,
            "kl_weight": self.kl_weight,
        }
