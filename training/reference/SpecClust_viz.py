"""
SpecClust – Section 3.1 Visualisations
=======================================
Produces:
  1. mirror_plots.pdf       – 6-panel grid: input (blue) vs reconstruction (orange)
  2. umap_category.pdf      – UMAP coloured by identified / novel / noise
  3. umap_charge.pdf        – UMAP coloured by precursor charge state (supplementary)

Usage:
    python specclust_viz.py \
        --parquet-dir  /path/to/cervical_cancer_parquets \
        --ae-dir       /path/to/ae_output \
        --cluster-dir  /path/to/cluster_output \
        --output-dir   figures

Requirements:
    pip install umap-learn matplotlib numpy pandas tensorflow

Notes:
    - ae-dir must contain ae_config.json + best_conditional_ae_weights.h5
    - cluster-dir must contain cluster_labels.npy + metadata.parquet
      (outputs of SpecCheckVSC.py cluster command).
      If not available, all unidentified spectra are labelled as "noise".
"""

import os
import gc
import glob
import json
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tensorflow as tf
from pathlib import Path

# ── Import from your own code ──────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SpecCheckVSC import (
    ConditionalSpectrumAutoencoder,
    Config,
    bin_spectrum_numpy,
)

config = Config()
MZ_MIN, MZ_MAX = config.MZ_MIN, config.MZ_MAX
N_BINS = config.N_BINS
BIN_SIZE = config.BIN_SIZE
MAX_CHARGE = config.MAX_CHARGE
MZ_AXIS = np.linspace(MZ_MIN + BIN_SIZE / 2, MZ_MAX - BIN_SIZE / 2, N_BINS)


# ════════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════════

def load_model(ae_dir: str) -> ConditionalSpectrumAutoencoder:
    """Build model from ae_config.json, then load weights."""
    config_path = os.path.join(ae_dir, "ae_config.json")
    with open(config_path) as f:
        ae_cfg = json.load(f)

    ae_cfg.pop("conditional", None)  # not a constructor arg

    model = ConditionalSpectrumAutoencoder(**ae_cfg)

    cond_dim = ae_cfg.get("cond_dim", MAX_CHARGE + 2)
    dummy_spec = tf.zeros((2, N_BINS))
    dummy_cond = tf.zeros((2, cond_dim))
    _ = model((dummy_spec, dummy_cond), training=False)

    # Support both weight-file naming conventions
    candidates = [
        os.path.join(ae_dir, "best_conditional_ae_weights.h5"),
        os.path.join(ae_dir, "best_conditional_ae.weights.h5"),
    ]
    weights_path = next((p for p in candidates if os.path.exists(p)), None)
    if weights_path is None:
        raise FileNotFoundError(
            f"Could not find weights in {ae_dir}. "
            f"Tried: {[os.path.basename(c) for c in candidates]}"
        )

    model.load_weights(weights_path)
    print(f"Loaded weights from {os.path.basename(weights_path)}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ════════════════════════════════════════════════════════════════════════════

def build_cond_vector(precursor_mz: float, charge: int,
                      ion_mobility: float) -> np.ndarray:
    """Build the 8-d conditioning vector matching training preprocessing."""
    charge_int = max(1, min(int(charge), MAX_CHARGE))
    charge_onehot = np.zeros(MAX_CHARGE, dtype=np.float32)
    charge_onehot[charge_int - 1] = 1.0
    mz_norm = float(precursor_mz) / config.PRECURSOR_MZ_MAX
    im_norm = float(np.clip(
        (ion_mobility - config.IM_MIN) / (config.IM_MAX - config.IM_MIN),
        0.0, 1.0
    ))
    return np.concatenate([charge_onehot, [mz_norm, im_norm]])


def load_sample_spectra(parquet_dir: str,
                        n_identified: int = 5000,
                        n_unidentified: int = 3000,
                        seed: int = 42) -> dict:
    """
    Stream parquet files and collect a balanced sample.
    Returns dict with keys:
        spectra      (N, 3200) float32
        conditioning (N, 8)    float32
        outcome      (N,)      int  [1=identified, 0=unidentified]
        peptide      (N,)      str
        charge       (N,)      int
        files        (N,)      str
    """
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    rng.shuffle(files)

    buckets = {1: [], 0: []}
    targets = {1: n_identified, 0: n_unidentified}

    for pf in files:
        if all(len(v) >= targets[k] for k, v in buckets.items()):
            break
        try:
            df = pd.read_parquet(pf).sample(frac=1, random_state=seed)
        except Exception as e:
            print(f"  Warning: {os.path.basename(pf)}: {e}")
            continue

        for _, row in df.iterrows():
            outcome = int(row.get("Outcome", 0))
            if len(buckets[outcome]) >= targets[outcome]:
                continue

            im = float(row.get("IM", row.get("IonMobility", 0.0)) or 0.0)
            if np.isnan(im) or im <= 0:
                continue

            mz_arr  = np.array(row["mz_array"], dtype=np.float32)
            int_arr = np.array(row["intensity_array"], dtype=np.float32)
            binned  = bin_spectrum_numpy(mz_arr, int_arr, config)

            pmz    = float(row.get("precursor_mz", row.get("PrecursorMz", 500.0)))
            charge = int(row.get("precursor_charge",
                         row.get("PrecursorCharge",
                         row.get("charge", 2))) or 2)
            cond   = build_cond_vector(pmz, charge, im)

            peptide = str(row.get("peptide", row.get("modified_peptide", "")))
            if peptide in ("nan", "None", ""):
                peptide = ""

            buckets[outcome].append({
                "spectrum":     binned,
                "conditioning": cond,
                "outcome":      outcome,
                "peptide":      peptide,
                "charge":       charge,
                "file":         os.path.basename(pf),
            })

        gc.collect()

    all_rows = buckets[1] + buckets[0]
    rng.shuffle(all_rows)

    return {
        "spectra":      np.array([r["spectrum"]     for r in all_rows], dtype=np.float32),
        "conditioning": np.array([r["conditioning"] for r in all_rows], dtype=np.float32),
        "outcome":      np.array([r["outcome"]      for r in all_rows], dtype=np.int32),
        "peptide":      [r["peptide"] for r in all_rows],
        "charge":       np.array([r["charge"]       for r in all_rows], dtype=np.int32),
        "files":        [r["file"] for r in all_rows],
    }


# ════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ════════════════════════════════════════════════════════════════════════════

def encode_batch(model, spectra: np.ndarray,
                 conditioning: np.ndarray,
                 batch_size: int = 256) -> np.ndarray:
    latents = []
    for i in range(0, len(spectra), batch_size):
        z = model.encode(
            tf.constant(spectra[i:i+batch_size]),
            tf.constant(conditioning[i:i+batch_size]),
            training=False,
        )
        if isinstance(z, tuple):
            z = z[0]
        latents.append(z.numpy())
    return np.concatenate(latents, axis=0)


def reconstruct_batch(model, spectra: np.ndarray,
                      conditioning: np.ndarray,
                      batch_size: int = 256) -> np.ndarray:
    recons = []
    for i in range(0, len(spectra), batch_size):
        r = model(
            (tf.constant(spectra[i:i+batch_size]),
             tf.constant(conditioning[i:i+batch_size])),
            training=False,
        )
        recons.append(r.numpy())
    return np.concatenate(recons, axis=0)


# ════════════════════════════════════════════════════════════════════════════
# 1. Mirror plots
# ════════════════════════════════════════════════════════════════════════════

def pick_mirror_examples(data: dict, n: int = 6) -> list:
    """
    Select n identified spectra with charge variety and rich peak content.
    """
    identified = np.where(data["outcome"] == 1)[0]
    spectra  = data["spectra"]
    charges  = data["charge"]

    selected = []
    for charge in [1, 2, 3, 4]:
        pool = [i for i in identified if charges[i] == charge]
        rich = [i for i in pool
                if np.sum(spectra[i] > 0.05 * spectra[i].max()) >= 8]
        if rich:
            selected.append(rich[0])

    remaining = [i for i in identified if i not in set(selected)]
    remaining.sort(key=lambda i: -np.sum(spectra[i] > 0.05 * spectra[i].max()))
    selected += remaining[:max(0, n - len(selected))]
    return selected[:n]


def plot_mirror_grid(model, data: dict, indices: list, savepath: str):
    n     = len(indices)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.0, nrows * 3.5),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    spectra  = data["spectra"][indices]
    cond     = data["conditioning"][indices]
    peptides = [data["peptide"][i] for i in indices]
    charges  = data["charge"][indices]
    recons   = reconstruct_batch(model, spectra, cond)

    for k, ax in enumerate(axes[:n]):
        inp = spectra[k]
        rec = recons[k]

        inp_norm = inp / (inp.max() + 1e-8)
        rec_norm = rec / (rec.max() + 1e-8)

        cos_sim = float(
            np.dot(inp_norm, rec_norm) /
            (np.linalg.norm(inp_norm) * np.linalg.norm(rec_norm) + 1e-8)
        )

        ax.vlines(MZ_AXIS,  0,  inp_norm, colors="#2166ac", linewidth=0.5,
                  label="Input")
        ax.vlines(MZ_AXIS,  0, -rec_norm, colors="#d6604d", linewidth=0.5,
                  label="Reconstruction")
        ax.axhline(0, color="k", linewidth=0.5)

        ax.set_ylim(-1.15, 1.15)
        ax.set_xlim(MZ_MIN, MZ_MAX)
        ax.set_xlabel("m/z", fontsize=8)
        ax.set_ylabel("Rel. intensity", fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=7)

        # Show absolute tick values
        locs = ax.get_yticks()
        ax.set_yticklabels([f"{abs(v):.1f}" for v in locs], fontsize=6)

        pep = (peptides[k][:22] if peptides[k] else "—")
        ax.set_title(f"{pep}  [+{charges[k]}]  Cosine similarity={cos_sim:.3f}",
                     fontsize=8, pad=3)

        if k == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.savefig(savepath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {savepath}")


# ════════════════════════════════════════════════════════════════════════════
# 2. UMAP
# ════════════════════════════════════════════════════════════════════════════

def build_umap_dataset(data: dict,
                       cluster_dir: str = None,
                       n_per_category: int = 3000,
                       seed: int = 42) -> dict:
    """
    Assign category labels and subsample.
    Categories: identified | novel | noise
    If cluster_dir is absent, unidentified spectra are all labelled 'noise'.
    """
    rng = np.random.RandomState(seed)
    categories = np.array([""] * len(data["outcome"]), dtype=object)
    categories[data["outcome"] == 1] = "identified"

    if cluster_dir and os.path.exists(
            os.path.join(cluster_dir, "cluster_labels.npy")):
        try:
            all_labels = np.load(os.path.join(cluster_dir, "cluster_labels.npy"))
            meta = pd.read_parquet(os.path.join(cluster_dir, "metadata.parquet"))
            meta["_mz_r"] = meta["precursor_mz"].round(2)
            meta["_key"]  = meta["file"] + "_" + meta["_mz_r"].astype(str)
            key_to_label  = dict(zip(meta["_key"], all_labels))

            for i in np.where(data["outcome"] == 0)[0]:
                mz_r = round(
                    float(data["conditioning"][i, MAX_CHARGE] * config.PRECURSOR_MZ_MAX),
                    2
                )
                key = f"{data['files'][i]}_{mz_r}"
                lbl = key_to_label.get(key, -2)
                categories[i] = "novel" if lbl >= 0 else "noise"

            print("  Category assignment: identified / novel / noise")
        except Exception as e:
            print(f"  Cluster alignment failed ({e}) – using identified/noise only")
            categories[data["outcome"] == 0] = "noise"
    else:
        categories[data["outcome"] == 0] = "noise"
        print("  No cluster_dir – unidentified labelled as 'noise'")

    cat_idx = {}
    for cat in ["identified", "novel", "noise"]:
        idx = np.where(categories == cat)[0]
        if len(idx) > n_per_category:
            idx = rng.choice(idx, size=n_per_category, replace=False)
        cat_idx[cat] = idx

    all_idx = np.sort(np.concatenate(list(cat_idx.values())))
    return {
        "spectra":      data["spectra"][all_idx],
        "conditioning": data["conditioning"][all_idx],
        "categories":   categories[all_idx],
        "charges":      data["charge"][all_idx],
    }


def run_umap(latents: np.ndarray, seed: int = 42) -> np.ndarray:
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("Run:  pip install umap-learn")
    print(f"  Running UMAP on {len(latents)} latents…")
    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="euclidean", random_state=seed, low_memory=True, verbose=False,
    )
    return reducer.fit_transform(latents)


def plot_umap_category(embedding, categories, savepath):
    STYLE = {
        "identified": dict(color="#2166ac", alpha=0.30, s=3, zorder=2,
                           label="Identified"),
        "novel":      dict(color="#d6604d", alpha=0.65, s=6, zorder=4,
                           label="Novel cluster"),
        "noise":      dict(color="#aaaaaa", alpha=0.15, s=2, zorder=1,
                           label="Noise / unclustered"),
    }
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cat in ["noise", "identified", "novel"]:
        mask = categories == cat
        if mask.any():
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       rasterized=True, **STYLE[cat])
    ax.legend(markerscale=3, framealpha=0.85, fontsize=9)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title("Latent space – spectrum category", fontsize=11)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(savepath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {savepath}")


def plot_umap_charge(embedding, charges, savepath):
    CHARGE_COLORS = {
        1: "#1b7837", 2: "#762a83", 3: "#d6604d",
        4: "#f4a582", 5: "#999999", 6: "#4d4d4d",
    }
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for charge in sorted(np.unique(charges)):
        mask = charges == charge
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=CHARGE_COLORS.get(int(charge), "#888888"),
                   alpha=0.30, s=3, rasterized=True, label=f"+{charge}")
    ax.legend(title="Charge", markerscale=3, framealpha=0.85, fontsize=9)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title("Latent space – precursor charge state", fontsize=11)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(savepath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {savepath}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SpecClust Section 3.1 visualisations"
    )
    parser.add_argument("--parquet-dir",    required=True)
    parser.add_argument("--ae-dir",         required=True)
    parser.add_argument("--cluster-dir",    default=None)
    parser.add_argument("--output-dir",     default="figures")
    parser.add_argument("--n-identified",   type=int, default=5000)
    parser.add_argument("--n-unidentified", type=int, default=3000)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n── Loading model ───────────────────────────────")
    model = load_model(args.ae_dir)

    print("\n── Loading sample spectra ──────────────────────")
    data = load_sample_spectra(
        args.parquet_dir,
        n_identified=args.n_identified,
        n_unidentified=args.n_unidentified,
        seed=args.seed,
    )
    print(f"  Identified: {int((data['outcome']==1).sum())}  "
          f"Unidentified: {int((data['outcome']==0).sum())}")

    print("\n── Mirror plots ────────────────────────────────")
    mirror_idx = pick_mirror_examples(data, n=6)
    plot_mirror_grid(model, data, mirror_idx, str(out / "mirror_plots.tiff"))

    print("\n── UMAP ────────────────────────────────────────")
    umap_data = build_umap_dataset(
        data, cluster_dir=args.cluster_dir, n_per_category=3000, seed=args.seed,
    )
    for cat in ["identified", "novel", "noise"]:
        n = int((umap_data["categories"] == cat).sum())
        if n:
            print(f"    {cat}: {n}")

    print("  Encoding latents…")
    latents = encode_batch(model, umap_data["spectra"], umap_data["conditioning"])
    embedding = run_umap(latents, seed=args.seed)

    plot_umap_category(embedding, umap_data["categories"],
                       str(out / "umap_category.tiff"))
    plot_umap_charge(embedding, umap_data["charges"],
                     str(out / "umap_charge.tiff"))

    print(f"\n── Done  ──  outputs in: {out}/")


if __name__ == "__main__":
    main()