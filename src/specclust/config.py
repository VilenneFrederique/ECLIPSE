"""Centralised configuration for SpecClust.

These values define the spectrum binning and the autoencoder architecture. They
MUST match the configuration used to train the published weights; the binning
parameters in particular are baked into the model input.
"""


class Config:
    """Centralized configuration."""

    SEED = 42

    # Binning parameters (autoencoder input)
    MZ_MIN = 100.0
    MZ_MAX = 1700.0
    BIN_SIZE = 0.5
    N_BINS = int((MZ_MAX - MZ_MIN) / BIN_SIZE)  # 3200

    # Preprocessing
    RELATIVE_INTENSITY_THRESHOLD = 0.01
    TOP_N_PEAKS = 100  # keep only the top N most intense peaks (None to disable)

    # Ion mobility normalisation (1/K0 range for peptides)
    IM_MIN = 0.6
    IM_MAX = 1.6

    # Precursor features
    MAX_CHARGE = 6
    PRECURSOR_MZ_MAX = 1700.0

    # Autoencoder architecture
    LATENT_DIM = 256
    AE_EMBED_DIM = 256
    AE_NUM_HEADS = 8
    AE_NUM_LAYERS = 4
    AE_FF_DIM = 512
    AE_PATCH_SIZE = 16
    AE_DROPOUT = 0.1

    # Conditioning embedding
    COND_EMBED_DIM = 256

    # Training - Autoencoder
    AE_BATCH_SIZE = 256
    AE_INITIAL_LR = 1e-4
    AE_EPOCHS = 50
    AE_WARMUP_EPOCHS = 5
    WEIGHT_DECAY = 1e-5


# Conditioning dimension: one-hot charge (MAX_CHARGE) + precursor_mz (1) + IM (1).
COND_DIM = Config.MAX_CHARGE + 2
