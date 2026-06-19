"""Embed spectra into the autoencoder latent space."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .config import Config
from .preprocessing import bin_spectrum_numpy, build_cond_vector


def embed_spectra(
    encoder,
    spectra: np.ndarray,
    conditioning: np.ndarray,
    batch_size: int = 256,
    latent_dim: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Encode pre-binned spectra to latent vectors in batches.

    Args:
        encoder: a loaded encoder (see :func:`eclipse_ms.modelhub.load_encoder`)
            or a full autoencoder exposing ``.encode``.
        spectra: array ``[n, N_BINS]`` of binned spectra (float32).
        conditioning: array ``[n, cond_dim]`` of conditioning vectors.
        batch_size: encode this many spectra at a time.
        latent_dim: output dimension; inferred from the first batch if None.

    Returns:
        Array ``[n, latent_dim]`` of latent vectors.
    """
    import tensorflow as tf

    n = len(spectra)
    if n != len(conditioning):
        raise ValueError(f"spectra and conditioning differ in length: {n} vs {len(conditioning)}")

    spectra = np.asarray(spectra, dtype=np.float32)
    conditioning = np.asarray(conditioning, dtype=np.float32)

    def _encode(x, cond):
        # Support both an Encoder (callable) and a full AE (has .encode).
        if hasattr(encoder, "encode"):
            z = encoder.encode(x, cond, training=False)
            if isinstance(z, (tuple, list)):
                z = z[0]
            return z
        return encoder((x, cond), training=False)

    if latent_dim is None:
        probe = _encode(tf.convert_to_tensor(spectra[:1]), tf.convert_to_tensor(conditioning[:1]))
        latent_dim = int(probe.shape[-1])

    latents = np.zeros((n, latent_dim), dtype=np.float32)
    start = time.time()
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        z = _encode(
            tf.convert_to_tensor(spectra[i:end]),
            tf.convert_to_tensor(conditioning[i:end]),
        )
        latents[i:end] = np.asarray(z)
        if verbose and (i // batch_size + 1) % 100 == 0:
            rate = end / (time.time() - start)
            print(f"  encoded {end:,}/{n:,} ({rate:.0f}/s)")
    if verbose:
        print(f"  done in {(time.time() - start) / 60:.1f} min")
    return latents


def embed_raw_spectra(
    encoder,
    mz_list,
    intensity_list,
    precursor_mz,
    charge,
    ion_mobility,
    config=Config,
    batch_size: int = 256,
    verbose: bool = True,
) -> np.ndarray:
    """Bin + condition raw peak lists, then embed.

    Each of ``precursor_mz``, ``charge``, ``ion_mobility`` is a sequence aligned
    with ``mz_list`` / ``intensity_list``.
    """
    n = len(mz_list)
    spectra = np.zeros((n, config.N_BINS), dtype=np.float32)
    cond = np.zeros((n, config.MAX_CHARGE + 2), dtype=np.float32)
    for i in range(n):
        spectra[i] = bin_spectrum_numpy(mz_list[i], intensity_list[i], config)
        cond[i] = build_cond_vector(precursor_mz[i], charge[i], ion_mobility[i], config)
    return embed_spectra(encoder, spectra, cond, batch_size=batch_size, verbose=verbose)
