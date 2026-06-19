# ECLIPSE

**ECLIPSE** embeds MS/MS spectra with a conditional transformer autoencoder
and clusters the latent space to discover candidate novel ("dark proteome")
peptides. Spectra are conditioned on precursor *m/z*, charge, and ion mobility,
so same-peptide spectra land close together in latent space.

The trained model is large (~650 MB), so it is **not** shipped on PyPI. The pip
package contains the code; the weights are hosted externally and downloaded +
cached on first use. Because embedding only needs the **encoder**, the default
download is encoder-only — roughly half the full model.

## Installation

```bash
pip install eclipse-ms                 # core: load model, embed, cluster
pip install "eclipse-ms[hdbscan]"      # add HDBSCAN clustering
pip install "eclipse-ms[viz]"          # add plotting (matplotlib, umap)
pip install "eclipse-ms[mzml]"         # add pyteomics for mzML
pip install "eclipse-ms[train]"        # everything needed to retrain
```

TensorFlow is a core dependency (the encoder needs it) but is imported lazily —
`import eclipse_ms` does not load TensorFlow until you actually build or load a
model.

## Getting the weights

`load_encoder()` resolves the model files in this order:

1. `ECLIPSE_MODEL_DIR` — set this to a folder holding the weights (handy on
   an HPC node where you already have them):
```bash
   export ECLIPSE_MODEL_DIR=/path/to/weights
```
2. the local cache (downloaded once, then reused);
3. download from the GitHub release (the URLs are configured in
   `eclipse_ms.modelhub.REGISTRY`).

With no setup, option 3 runs automatically on first use. You can also bypass
the registry with explicit local paths:

```python
from eclipse_ms import load_encoder
encoder = load_encoder(weights="encoder.weights.h5", config="encoder_config.json")
```

## Quick start

```python
import numpy as np
from eclipse_ms import load_encoder, embed_raw_spectra, cluster_latents, score_clusters

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
eclipse embed   -i parquet_dir/ -o latents.npy
eclipse cluster -i latents.npy  -o clusters/ --method hdbscan
```

## Repository layout

```
src/eclipse_ms/     installable package (model, embed, cluster, consensus, CLI)
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
