"""ECLIPSE command-line interface.

Subcommands:
  embed      Bin + encode spectra from parquet to a latents .npy
  cluster    Cluster a latents .npy into cluster labels
  consensus  Build consensus spectra (mzML) from clusters

Training and HPC data-prep live in the repo's ``training/`` scripts, not in the
installed package.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _cmd_embed(args):
    import pandas as pd

    from .config import Config
    from .embed import embed_raw_spectra
    from .modelhub import load_encoder

    encoder = load_encoder(weights=args.weights, config=args.config)

    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    print(f"Found {len(files)} parquet files")

    mz, inten, pmz, charge, im = [], [], [], [], []
    for fp in files:
        df = pd.read_parquet(fp)
        for _, row in df.iterrows():
            mz.append(row["mz_array"])
            inten.append(row["intensity_array"])
            pmz.append(float(row.get("precursor_mz", 0.0)))
            charge.append(int(row.get("precursor_charge", 2)))
            im.append(float(row.get("ion_mobility", 0.0)))
            if args.max_spectra and len(mz) >= args.max_spectra:
                break
        if args.max_spectra and len(mz) >= args.max_spectra:
            break

    latents = embed_raw_spectra(encoder, mz, inten, pmz, charge, im, Config, args.batch_size)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, latents)
    print(f"Saved {latents.shape} latents to {args.output}")


def _cmd_cluster(args):
    from .cluster import cluster_latents, score_clusters

    latents = np.load(args.input)
    labels, info = cluster_latents(
        latents, method=args.method, min_cluster_size=args.min_cluster_size
    )
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "cluster_labels.npy"), labels)
    with open(os.path.join(args.output, "cluster_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    scores = score_clusters(latents, labels)
    scores.to_csv(os.path.join(args.output, "cluster_scores.csv"), index=False)
    print(f"Clusters: {info.get('n_clusters')}, noise: {info.get('n_noise', 0)}")
    print(f"Wrote labels, info, and scores to {args.output}")


def _cmd_consensus(args):
    print(
        "Consensus generation needs spectra grouped by cluster. See "
        "eclipse_ms.consensus.generate_consensus_spectrum and the example in "
        "training/consensus_reference.py for the full pipeline."
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="eclipse", description="ECLIPSE")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("embed", help="Encode spectra to latents")
    pe.add_argument("-i", "--input", required=True, help="Parquet directory")
    pe.add_argument("-o", "--output", required=True, help="Output latents .npy")
    pe.add_argument("--weights", default=None, help="Local encoder weights (.h5)")
    pe.add_argument("--config", default=None, help="Local encoder config (.json)")
    pe.add_argument("--batch-size", type=int, default=256)
    pe.add_argument("--max-spectra", type=int, default=None)
    pe.set_defaults(func=_cmd_embed)

    pc = sub.add_parser("cluster", help="Cluster latents")
    pc.add_argument("-i", "--input", required=True, help="latents .npy")
    pc.add_argument("-o", "--output", required=True, help="Output directory")
    pc.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan")
    pc.add_argument("--min-cluster-size", type=int, default=5)
    pc.set_defaults(func=_cmd_cluster)

    pk = sub.add_parser("consensus", help="Consensus spectra from clusters")
    pk.set_defaults(func=_cmd_consensus)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
