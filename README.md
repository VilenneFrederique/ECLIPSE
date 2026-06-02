# SpecClust

**SpecClust** embeds MS/MS spectra with a conditional transformer autoencoder
and clusters the latent space to discover candidate novel ("dark proteome")
peptides. Spectra are conditioned on precursor *m/z*, charge, and ion mobility,
so same-peptide spectra land close together in latent space.

The trained model is large (~650 MB), so it is **not** shipped on PyPI. The pip
package contains the code; the weights are hosted externally and downloaded +
cached on first use. Because embedding only needs the **encoder**, the default
download is encoder-only — roughly half the full model.

## Installation

```bash
pip install specclust                 # core: load model, embed, cluster
pip install "specclust[hdbscan]"      # add HDBSCAN clustering
pip install "specclust[viz]"          # add plotting (matplotlib, umap)
pip install "specclust[mzml]"         # add pyteomics for mzML
pip install "specclust[train]"        # everything needed to retrain
```

TensorFlow is a core dependency (the encoder needs it) but is imported lazily —
`import specclust` does not load TensorFlow until you actually build or load a
model.

## Getting the weights

`load_encoder()` resolves the model files in this order:

1. `SPECCLUST_MODEL_DIR` — set this to a folder holding the weights (handy on
   an HPC node where you already have them):
```bash
   export SPECCLUST_MODEL_DIR=/path/to/weights
```
2. the local cache (downloaded once, then reused);
3. download from the GitHub release (the URLs are configured in
   `specclust.modelhub.REGISTRY`).

With no setup, option 3 runs automatically on first use. You can also bypass
the registry with explicit local paths:

```python
from specclust import load_encoder
encoder = load_encoder(weights="encoder.weights.h5", config="encoder_config.json")
```

## Quick start

```python
import numpy as np
from specclust import load_encoder, embed_raw_spectra, cluster_latents, score_clusters

encoder = load_encoder()  # downloads/caches the encoder on first call

# raw peak lists + precursor info (lists aligned by index)
latents = embed_raw_spectra(
    encoder,
    mz_list, intensity_list,
    precursor_mz, charge, ion_mobility,
)

labels, info = cluster_latents(latents, method="hdbscan", min_cluster_size=5)
scores = score_clusters(latents, labels)
print(info["n_clusters"], "clusters")
```

Command line:

```bash
specclust embed   -i parquet_dir/ -o latents.npy
specclust cluster -i latents.npy  -o clusters/ --method hdbscan
```

## Repository layout

```
src/specclust/      installable package (model, embed, cluster, consensus, CLI)
training/           NOT installed: HPC data-prep, training, and export scripts
  export_encoder.py   run once to create the slim encoder assets to publish
  reference/          the original monolithic scripts, kept verbatim
tests/              run without TensorFlow or weights (model tests auto-skip)
```

## Citation
Publication Pending!

Vilenne, Frédérique & Valkenborg Dirk. Clustering the Dark Proteome: A Deep Learning Approach to Novel Peptide Discovery in Immunopeptidomics (2026)

## License

MIT © 2026 Frédérique Vilenne, Dirk Valkenborg
