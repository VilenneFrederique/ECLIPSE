# Training & data preparation (reference)

These scripts are **not** part of the installed `eclipse_ms` package. They are
the original HPC pipeline, kept for reproducibility, plus the one step you run
to publish the model.

- `export_encoder.py` — **run this once** to produce the slim
  `specclust_encoder.weights.h5` + `encoder_config.json` from a trained
  autoencoder. Upload those to your release; `eclipse_ms` downloads them on
  first use. (Imports from the installed `eclipse_ms` package.)
- `reference/SpecCheckVSC.py` — the original monolith: model, training
  (`train-conditional-ae`), clustering, and visualisation CLI.
- `reference/SpecCheckDataPrepVSC.py` — mzML/FragPipe → parquet → TFRecord data
  prep. Contains cluster-specific absolute paths (`/lustre1/scratch/...`); edit
  for your environment.
- `reference/consensus_reference.py` — full consensus-spectrum + mzML writer.
- `reference/SpecClust_viz.py` — original visualisation script.

To retrain, install the training extra:

```bash
pip install -e ".[train,hdbscan]"
```

The model architecture used by the package (`eclipse_ms.models`) is identical to
the one in the reference monolith, so weights trained there load directly.
