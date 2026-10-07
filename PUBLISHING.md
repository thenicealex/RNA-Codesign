# Weight release

This standalone source snapshot uses Apache-2.0, with third-party notices and
license texts included. The parent research repository and its Git history are
outside the release. Model checkpoints are intentionally excluded.

When publishing weights:

1. Select a compatible schema-v2 checkpoint and export it with
   `src/export_checkpoint.py`. The exporter keeps raw/available EMA weights and
   model architecture while omitting training data paths and optimizer state.
2. Run all four README examples on a GPU compute node using the exported
   checkpoint. Evaluate the RhoFold-derived public geometry templates, which
   differ from the research snapshot's original tables.
3. Choose the weight distribution terms and upload the exported checkpoint to
   the chosen release location. Add its download URL and checksum to the README,
   and update the validation statement to reflect the actual results.

Do not commit datasets, research outputs, credentials, or checkpoint files.
