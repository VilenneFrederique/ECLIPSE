"""Model registry, download/cache, and loaders.

The trained weights are far too large to ship inside the PyPI wheel, so they
live in external storage (a GitHub Release asset, a Hugging Face Hub file, or a
Zenodo record) and are downloaded on first use and cached locally, with a
SHA-256 integrity check.

Resolution order for any model file:
  1. ``ECLIPSE_MODEL_DIR`` env var, if set and the file exists there;
  2. the local cache (``platformdirs`` user cache dir);
  3. download from the registry URL into the cache.

You can also bypass the registry entirely and pass explicit local paths to
:func:`load_encoder` / :func:`load_autoencoder` (e.g. on an HPC node where you
already have the weights).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_dir

# ---------------------------------------------------------------------------
# Registry. After uploading your weights, fill in `url` and `sha256` for each
# entry. `sha256=None` disables the integrity check (not recommended for a
# release). Compute a hash with:  python -c "import hashlib,sys;
# print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" FILE
# ---------------------------------------------------------------------------
REGISTRY: dict[str, dict] = {
    # Slim, recommended for embedding/clustering: encoder weights only (~half size).
    "encoder-weights": {
        "filename": "specclust_encoder.weights.h5",
        "url": "https://github.com/VilenneFrederique/ECLIPSE/releases/download/v0.1.0/specclust_encoder.weights.h5",
        "sha256": "3c90bb9bb5c9960251f9b2165dd61be89f5ed78be6b3d21f5d28a0bd49877a6e",
    },
    "encoder-config": {
        "filename": "encoder_config.json",
        "url": "https://github.com/VilenneFrederique/ECLIPSE/releases/download/v0.1.0/encoder_config.json",
        "sha256": "89e53685f735458973c358746fb5444148cc6813725d93d7a45fcfd9974c0a00",
    },
}


def cache_dir() -> Path:
    """Directory where downloaded weights are cached."""
    d = Path(user_cache_dir("eclipse-ms"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    if url in (None, "", "REPLACE_ME"):
        raise RuntimeError(
            f"No download URL configured for {dest.name}. Either set the URL in "
            f"eclipse_ms.modelhub.REGISTRY, set the ECLIPSE_MODEL_DIR environment "
            f"variable to a folder containing the file, or pass an explicit path."
        )
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} from {url} ...")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r  {read / 1e6:,.0f} / {total / 1e6:,.0f} MB ({pct:.0f}%)", end="")
    print()
    tmp.replace(dest)


def get_model_file(key: str) -> Path:
    """Resolve a registry key to a local path, downloading/caching as needed."""
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key '{key}'. Known: {list(REGISTRY)}")
    entry = REGISTRY[key]
    filename = entry["filename"]

    env_dir = os.environ.get("ECLIPSE_MODEL_DIR")
    if env_dir:
        candidate = Path(env_dir) / filename
        if candidate.exists():
            return candidate

    cached = cache_dir() / filename
    if cached.exists():
        if entry.get("sha256") and _sha256(cached) != entry["sha256"]:
            print(f"Cached {filename} failed checksum; re-downloading.")
            cached.unlink()
        else:
            return cached

    _download(entry["url"], cached)
    if entry.get("sha256") and _sha256(cached) != entry["sha256"]:
        cached.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {filename} after download.")
    return cached


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _build_and_load_encoder(config: dict, weights_path: str):
    import tensorflow as tf

    from .config import COND_DIM
    from .models import ConditionalSpectrumEncoder

    cfg = {k: v for k, v in config.items() if k not in ("conditional", "use_kl", "kl_weight")}
    encoder = ConditionalSpectrumEncoder(**cfg)

    cond_dim = config.get("cond_dim", COND_DIM)
    n_bins = config.get("n_bins", 3200)
    _ = encoder((tf.zeros((2, n_bins)), tf.zeros((2, cond_dim))), training=False)
    encoder.load_weights(weights_path)
    return encoder


def load_encoder(weights: Optional[str] = None, config: Optional[str] = None):
    """Load the encoder for embedding spectra.

    With no arguments, downloads/caches the published encoder weights. Pass
    explicit ``weights`` (``.h5``) and ``config`` (``.json``) paths to load a
    local model instead.
    """
    weights_path = weights or str(get_model_file("encoder-weights"))
    config_path = config or str(get_model_file("encoder-config"))
    with open(config_path) as f:
        cfg = json.load(f)
    return _build_and_load_encoder(cfg, weights_path)


def load_autoencoder(weights: Optional[str] = None, config: Optional[str] = None):
    """Load the full autoencoder (encoder + decoder).

    Needed only for reconstruction / visualisation; embedding and clustering
    use :func:`load_encoder`, which is roughly half the download.
    """
    import tensorflow as tf

    from .config import COND_DIM
    from .models import ConditionalSpectrumAutoencoder

    weights_path = weights or str(get_model_file("ae-weights"))
    config_path = config or str(get_model_file("ae-config"))
    with open(config_path) as f:
        cfg = json.load(f)

    ctor = {k: v for k, v in cfg.items() if k != "conditional"}
    ae = ConditionalSpectrumAutoencoder(**ctor)
    cond_dim = cfg.get("cond_dim", COND_DIM)
    n_bins = cfg.get("n_bins", 3200)
    _ = ae((tf.zeros((2, n_bins)), tf.zeros((2, cond_dim))), training=False)
    ae.load_weights(weights_path)
    return ae
