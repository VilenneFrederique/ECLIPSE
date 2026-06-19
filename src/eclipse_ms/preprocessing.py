"""Spectrum preprocessing (NumPy).

These reproduce the exact binning and conditioning used during training, so
embeddings computed at inference time match the model's expectations. The
NumPy implementation keeps this module importable without TensorFlow.
"""

from __future__ import annotations

import numpy as np

from .config import Config


def bin_spectrum_numpy(mz: np.ndarray, intensity: np.ndarray, config=Config) -> np.ndarray:
    """Bin a single spectrum to the fixed-width vector the encoder expects."""
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    mask = (mz >= config.MZ_MIN) & (mz < config.MZ_MAX) & (intensity > 0)
    mz = mz[mask]
    intensity = intensity[mask]
    if len(intensity) == 0:
        return np.zeros(config.N_BINS, dtype=np.float32)

    intensity = intensity / intensity.max()

    mask = intensity >= config.RELATIVE_INTENSITY_THRESHOLD
    mz = mz[mask]
    intensity = intensity[mask]
    if len(intensity) == 0:
        return np.zeros(config.N_BINS, dtype=np.float32)

    if getattr(config, "TOP_N_PEAKS", None) and len(intensity) > config.TOP_N_PEAKS:
        top_idx = np.argsort(intensity)[-config.TOP_N_PEAKS:]
        mz = mz[top_idx]
        intensity = intensity[top_idx]

    intensity = np.sqrt(intensity)

    bin_indices = ((mz - config.MZ_MIN) / config.BIN_SIZE).astype(int)
    bin_indices = np.clip(bin_indices, 0, config.N_BINS - 1)

    binned = np.zeros(config.N_BINS, dtype=np.float32)
    np.maximum.at(binned, bin_indices, intensity.astype(np.float32))

    if binned.max() > 0:
        binned = binned / binned.max()

    return binned


def build_cond_vector(
    precursor_mz: float,
    charge: int,
    ion_mobility: float,
    config=Config,
) -> np.ndarray:
    """Build the conditioning vector (one-hot charge + norm. m/z + norm. IM).

    Matches the training preprocessing; length is ``config.MAX_CHARGE + 2``.
    """
    charge_int = max(1, min(int(charge), config.MAX_CHARGE))
    charge_onehot = np.zeros(config.MAX_CHARGE, dtype=np.float32)
    charge_onehot[charge_int - 1] = 1.0

    mz_norm = float(precursor_mz) / config.PRECURSOR_MZ_MAX
    im_norm = float(
        np.clip((ion_mobility - config.IM_MIN) / (config.IM_MAX - config.IM_MIN), 0.0, 1.0)
    )
    return np.concatenate([charge_onehot, [mz_norm, im_norm]]).astype(np.float32)


def preprocess(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    charge: int,
    ion_mobility: float,
    config=Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: return ``(binned_spectrum, conditioning_vector)``."""
    binned = bin_spectrum_numpy(mz, intensity, config)
    cond = build_cond_vector(precursor_mz, charge, ion_mobility, config)
    return binned, cond
