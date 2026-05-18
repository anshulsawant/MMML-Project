Archived material that is not part of the active v12 CoD manifold-anchoring pipeline.

- `baselines/`: side-project baseline folders kept for reference.
- `legacy/`: deprecated experiment code, dynamic HALT prototypes, and one-off training scripts.
- `logs/`: historical batch-generation logs and recovery artifacts.
- `results/`: old exploratory outputs kept for paper support.

The active repository surface is now centered on:

- `configs/v12_cod.yaml`
- `data/build_manifold.py`
- `training/train_x_encoder.py`
- `training/train_decoder_projection.py`
- `training/train_manifold_anchor.py`
- `eval/e2e.py`
- `scripts/visualize.py`