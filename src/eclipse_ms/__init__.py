"""ECLIPSE: conditional spectrum autoencoder + clustering for MS/MS.

Importing this package does NOT import TensorFlow; TF is loaded lazily the
first time you build or load a model (e.g. ``load_encoder``) or call
``embed_spectra``. This keeps imports fast and lets the clustering / consensus
utilities be used in TF-free environments.

The Keras model classes live in ``eclipse_ms.models`` (importing that submodule
does import TensorFlow).
"""

from .config import COND_DIM, Config
from .preprocessing import bin_spectrum_numpy, build_cond_vector, preprocess
from .cluster import cluster_latents, score_clusters
from .consensus import generate_consensus_spectrum, write_mzml
from .modelhub import (
    REGISTRY,
    cache_dir,
    get_model_file,
    load_autoencoder,
    load_encoder,
)
from .embed import embed_raw_spectra, embed_spectra

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "COND_DIM",
    "bin_spectrum_numpy",
    "build_cond_vector",
    "preprocess",
    "load_encoder",
    "load_autoencoder",
    "get_model_file",
    "cache_dir",
    "REGISTRY",
    "embed_spectra",
    "embed_raw_spectra",
    "cluster_latents",
    "score_clusters",
    "generate_consensus_spectrum",
    "write_mzml",
]
