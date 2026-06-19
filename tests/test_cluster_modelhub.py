import os

import numpy as np
import pytest

from eclipse_ms import cluster_latents, score_clusters
from eclipse_ms import modelhub


def _three_blobs(n=300, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[5.0] * dim, [-5.0] * dim, [0.0] * dim])
    pts = np.vstack([c + rng.normal(scale=0.3, size=(n, dim)) for c in centers])
    return pts.astype(np.float32)


def test_kmeans_recovers_blobs():
    latents = _three_blobs()
    labels, info = cluster_latents(latents, method="kmeans", pca_dims=8)
    assert info["method"] == "kmeans"
    assert len(labels) == len(latents)
    assert info["n_clusters"] >= 3


def test_score_clusters_columns_and_order():
    latents = _three_blobs()
    labels, _ = cluster_latents(latents, method="kmeans", pca_dims=8)
    scores = score_clusters(latents, labels)
    assert set(["cluster", "size", "cohesion_mean_dist"]).issubset(scores.columns)
    # sorted by size descending
    assert list(scores["size"]) == sorted(scores["size"], reverse=True)


def test_registry_shape():
    for key, entry in modelhub.REGISTRY.items():
        assert "filename" in entry and "url" in entry and "sha256" in entry


def test_env_dir_resolution(tmp_path, monkeypatch):
    # Place a file where ECLIPSE_MODEL_DIR points; it should resolve without download.
    fname = modelhub.REGISTRY["encoder-config"]["filename"]
    (tmp_path / fname).write_text("{}")
    monkeypatch.setenv("ECLIPSE_MODEL_DIR", str(tmp_path))
    path = modelhub.get_model_file("encoder-config")
    assert os.path.basename(path) == fname
    assert str(tmp_path) in str(path)


def test_missing_url_raises(monkeypatch, tmp_path):
    # No env dir, empty cache, placeholder URL -> informative error.
    monkeypatch.delenv("ECLIPSE_MODEL_DIR", raising=False)
    monkeypatch.setattr(modelhub, "cache_dir", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="No download URL"):
        modelhub.get_model_file("encoder-weights")
