"""Export encoder-only weights from a trained autoencoder.

Embedding and clustering need only the encoder. Splitting the encoder out of
the full autoencoder roughly halves the download users have to fetch and lets
you publish a slim `specclust_encoder.weights.h5` + `encoder_config.json`
alongside (or instead of) the full model.

Run once, where the trained autoencoder weights are available:

    python training/export_encoder.py \
        --ae-dir /path/to/autoencoder_dir \
        --out    /path/to/release_assets

`--ae-dir` must contain `ae_config.json` and `best_conditional_ae.weights.h5`.
After running, verify the encoder reproduces the full model's latents on a few
spectra before publishing (a check is sketched at the bottom).
"""

import argparse
import json
import os

import tensorflow as tf

from specclust.config import COND_DIM
from specclust.models import ConditionalSpectrumAutoencoder


def main():
    ap = argparse.ArgumentParser(description="Export encoder-only weights")
    ap.add_argument("--ae-dir", required=True, help="Dir with ae_config.json + AE weights")
    ap.add_argument("--out", required=True, help="Output dir for encoder assets")
    args = ap.parse_args()

    with open(os.path.join(args.ae_dir, "ae_config.json")) as f:
        ae_cfg = json.load(f)

    ctor = {k: v for k, v in ae_cfg.items() if k != "conditional"}
    ae = ConditionalSpectrumAutoencoder(**ctor)

    cond_dim = ae_cfg.get("cond_dim", COND_DIM)
    n_bins = ae_cfg.get("n_bins", 3200)
    _ = ae((tf.zeros((2, n_bins)), tf.zeros((2, cond_dim))), training=False)
    ae.load_weights(os.path.join(args.ae_dir, "best_conditional_ae.weights.h5"))
    print("Loaded full autoencoder.")

    os.makedirs(args.out, exist_ok=True)

    # Save encoder weights only.
    enc_weights = os.path.join(args.out, "specclust_encoder.weights.h5")
    ae.encoder.save_weights(enc_weights)

    # Write a COMPLETE encoder config (all constructor args), so the encoder
    # reconstructs exactly regardless of the slim get_config() defaults.
    enc_cfg = {
        "n_bins": ae_cfg.get("n_bins", 3200),
        "patch_size": ae_cfg.get("patch_size", 16),
        "embed_dim": ae_cfg.get("embed_dim", 256),
        "num_heads": ae_cfg.get("num_heads", 8),
        "num_layers": ae_cfg.get("num_layers", 4),
        "ff_dim": ae_cfg.get("ff_dim", 512),
        # If the AE used KL, the encoder's latent_dim is doubled internally.
        "latent_dim": ae_cfg.get("latent_dim", 256)
        * (2 if ae_cfg.get("use_kl") else 1),
        "cond_dim": cond_dim,
        "dropout": ae_cfg.get("dropout", 0.1),
    }
    with open(os.path.join(args.out, "encoder_config.json"), "w") as f:
        json.dump(enc_cfg, f, indent=2)

    size_mb = os.path.getsize(enc_weights) / 1e6
    print(f"Wrote {enc_weights} ({size_mb:,.1f} MB) and encoder_config.json")

    # ---- SANITY CHECK (uncomment): encoder latents must match the AE ----
    # import numpy as np
    # from specclust.modelhub import load_encoder
    # enc = load_encoder(weights=enc_weights,
    #                    config=os.path.join(args.out, "encoder_config.json"))
    # x = tf.random.uniform((4, n_bins)); c = tf.random.uniform((4, cond_dim))
    # z_ae  = ae.encode(x, c, training=False)
    # z_enc = enc((x, c), training=False)
    # print("max abs diff:", float(tf.reduce_max(tf.abs(z_ae - z_enc))))  # ~0


if __name__ == "__main__":
    main()
