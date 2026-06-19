import numpy as np
import pytest

# Skip the whole module if TensorFlow is not installed.
tf = pytest.importorskip("tensorflow")

from eclipse_ms.config import COND_DIM, Config  # noqa: E402
from eclipse_ms.models import (  # noqa: E402
    ConditionalSpectrumAutoencoder,
    ConditionalSpectrumEncoder,
)


def test_encoder_output_shape():
    enc = ConditionalSpectrumEncoder(
        n_bins=Config.N_BINS, latent_dim=Config.LATENT_DIM, cond_dim=COND_DIM
    )
    x = tf.zeros((4, Config.N_BINS))
    cond = tf.zeros((4, COND_DIM))
    z = enc((x, cond), training=False)
    assert tuple(z.shape) == (4, Config.LATENT_DIM)


def test_autoencoder_roundtrip_shapes():
    ae = ConditionalSpectrumAutoencoder(
        n_bins=Config.N_BINS, latent_dim=Config.LATENT_DIM, cond_dim=COND_DIM
    )
    x = tf.zeros((2, Config.N_BINS))
    cond = tf.zeros((2, COND_DIM))
    recon = ae((x, cond), training=False)
    assert tuple(recon.shape) == (2, Config.N_BINS)
    z = ae.encode(x, cond, training=False)
    assert tuple(z.shape) == (2, Config.LATENT_DIM)


def test_embed_spectra_runs():
    from eclipse_ms.embed import embed_spectra

    enc = ConditionalSpectrumEncoder(
        n_bins=Config.N_BINS, latent_dim=Config.LATENT_DIM, cond_dim=COND_DIM
    )
    spectra = np.zeros((5, Config.N_BINS), dtype=np.float32)
    cond = np.zeros((5, COND_DIM), dtype=np.float32)
    latents = embed_spectra(enc, spectra, cond, batch_size=2, verbose=False)
    assert latents.shape == (5, Config.LATENT_DIM)
