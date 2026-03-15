# LatentEuclid Experiment Log (Report 3)

## Experiment Tracking Protocol
This document coordinates the ablations and intrinsic metrics tested on the LatentEuclid pipeline to support project reporting. Each experiment modifies specific variables across the 4 primary phases:
1. Target Manifold Generation (`build_manifold.py`)
2. Continuous Teacher Topologies (`train_x_encoder.py`)
3. Discrete Target Projection (`train_decoder_projection.py`)
4. Answer Extraction/Evaluation (`eval_e2e.py`)

All artifacts for an experiment map natively to its canonical `experiment_name`.

---

## 1. `v1_vicreg_baseline` (Baseline Control)
**Hypothesis & Modifications:**
- The legacy Baseline execution of LatentEuclid utilizing zero custom modifications.
- **Manifold Construction:** Extracted entirely from the raw `<eos>` end-token hidden state of the inference step sequence (anisotropic bottleneck).
- **Teacher Loss Formulation:** Handled via vanilla `VICReg`, which artificially scales predictor magnitudes to force a pseudo-variance rather than preserving true representation magnitudes.

**Failure Modes & Results:**
- **Zero-Shot Accuracy:** 27.5% *(Note: Discovered this was actually a "blind" text-only run due to a dataloader bug that completely dropped image inputs. The language model guessed answers based heavily on leaked deterministic math in the text prompts).*
- Continuous magnitudes shattered LLM LayerNorm stability upon projection.

---

## 2. `v2_huber_mean_pooled`
**Hypothesis & Modifications:**
- Addressing the magnitude stretching contradiction in Phase 2 by reverting strictly to Supervised Alignment Loss (Huber + Cosine).
- Addressing the anisotropic `<eos>` bottleneck in Phase 3 by computing Attention-Weighted Mean Pooling across the entire 4096-dimensional hidden state sequence.
- **Manifold Construction:** `target_tensors_v2_mean_pooled`
- **Teacher Loss Formulation:** `huber_cosine` (Cosine weighted 10x)
- **Micro-Batch Size:** 32 (Gradient Checkpointing Enabled)
- **Gradient Accumulation:** 4 (Global Batch Size: 128)
- **Throughput:** ~75s per global step on A100.

**Failure Modes & Results:**
- **Zero-Shot E2E Accuracy:** 38.15% (Up from 27.5% baseline)
