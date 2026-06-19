"""Cluster latent vectors and score the resulting clusters."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def cluster_latents(
    latents: np.ndarray,
    method: str = "hdbscan",
    min_cluster_size: int = 5,
    pca_dims: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, dict]:
    """Cluster latent vectors with HDBSCAN (preferred) or MiniBatchKMeans.

    Reduces dimensionality with PCA first when ``n_dims > pca_dims``. Falls back
    to KMeans if HDBSCAN is not installed.

    Returns ``(labels, info)`` where ``labels`` is ``-1`` for HDBSCAN noise.
    """
    from sklearn.decomposition import PCA

    n_samples, n_dims = latents.shape
    info = {"method": method, "n_samples": int(n_samples)}

    if n_dims > pca_dims:
        pca = PCA(n_components=pca_dims, random_state=random_state)
        reduced = pca.fit_transform(latents)
        info["pca_variance_explained"] = float(pca.explained_variance_ratio_.sum())
    else:
        reduced = latents

    if method == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            print("hdbscan not installed (`pip install eclipse-ms[hdbscan]`); using KMeans.")
            method = "kmeans"

    if method == "hdbscan":
        import hdbscan

        start = time.time()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=3,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(reduced)
        info["time"] = time.time() - start
        info["n_clusters"] = len(set(labels)) - (1 if -1 in labels else 0)
        info["n_noise"] = int((labels == -1).sum())

    elif method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = max(2, min(10000, n_samples // 10))
        start = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=1024, random_state=random_state, n_init=3
        )
        labels = kmeans.fit_predict(reduced)
        info["time"] = time.time() - start
        info["n_clusters"] = len(set(labels))
        info["n_noise"] = 0
    else:
        raise ValueError(f"Unknown method: {method}")

    return labels, info


def score_clusters(latents: np.ndarray, labels: np.ndarray) -> "pd.DataFrame":  # noqa: F821
    """Lightweight per-cluster quality scores from latent geometry.

    For each non-noise cluster, reports size and intra-cluster cohesion
    (mean distance to centroid; smaller = tighter).
    """
    import pandas as pd

    rows = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        pts = latents[idx]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        rows.append(
            {
                "cluster": int(c),
                "size": int(len(idx)),
                "cohesion_mean_dist": float(dists.mean()),
                "cohesion_std_dist": float(dists.std()),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("size", ascending=False).reset_index(drop=True)
    return df
