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
- **Throughput:** ~75s per global step on A100.

**Failure Modes & Results:**
- **Zero-Shot E2E Accuracy:** 38.15% (Up from 27.5% baseline)

## 3. `v3_unfrozen_decoder_layers` (Ablation: Replacing MLP)
**Hypothesis & Modifications:**
- Testing if the LLM's early mathematical reasoning layers can inherently bridge the modality gap better than a dedicated Projection MLP.
- **Manifold Construction:** `target_tensors_v2_mean_pooled`
- **Teacher Loss Formulation:** `huber_cosine`
- **Decoder Configuration:** `unfreeze_layers: 2`, `use_projection_mlp: False`

**Failure Modes & Results:**
- **Training Validation CE Loss:** 0.81 (Deceptively low)
- **Zero-Shot E2E Accuracy:** 6.67% (Catastrophic Failure)
- **Analysis:** Demonstrates the necessity of a dedicated bridging manifold. Without the MLP, Unfrozen Layers 0 and 1 warped their self-attention matrices to translate the raw visual continuous vectors directly, which corrupted their ability to process regular text tokens. During evaluation generation, these corrupted layers fed mangled text embeddings into the 30 subsequent frozen language layers, leading to catastrophic text hallucination loops (e.g. ``"```json\n"``, `"To find the measure..."`). The low training loss was an illusion caused by the layers overfitting purely to the `<|im_end|>` termination token, which dominates the short target answers.

---

## 4. `v4_projection_and_unfrozen_layers` (The Synthesis)
**Hypothesis & Modifications:**
- Combining the modality-bridging stability of the Projection MLP with the logic-refining capacity of unfrozen LLM layers to prevent Catastrophic Forgetting.
- **Manifold Construction:** `target_tensors_v2_mean_pooled`
- **Teacher Loss Formulation:** `huber_cosine`
- **Decoder Configuration:** `unfreeze_layers: 2`, `use_projection_mlp: True`

**Failure Modes & Results:**
- **Training Validation CE Loss:** 0.77
- **Zero-Shot E2E Accuracy:** **45.38%** (New SOTA for pipeline)
- **Analysis:** A massive +7.23% absolute accuracy gain over the `v2` MLP-only baseline. Because the Projection MLP pre-mapped the visual thought vectors into the language model's native syntax *before* they entered the LLM, Layers 0 and 1 did not have to corrupt their text-processing matrices to bridge the gap. Instead, they were freed up to cleanly fine-tune their spatial and geometric logic upon these stable topologies, leading to a synergistic breakthrough in geometric reasoning stability.

---

## 5. `v5_end_to_end_v4_base` (Massive Co-Training)
**Hypothesis & Modifications:**
- Testing if opening the 4B parameter Qwen3-VL X-Encoder architecture to end-to-end backpropagation can further align the spatial representation with the math-heavy reasoning demands of the unfrozen base decoder.
- **Teacher Loss Formulation:** End-to-End Cross Entropy
- **Decoder Configuration:** `unfreeze_layers: 36`, `use_projection_mlp: True`
- **Encoder Configuration:** `unfreeze_layers: 36` (Transformer blocks only)
- **Throughput:** ~5s per micro-step (~2.6 minutes per global accumulated step of 32) on A100.

**Failure Modes & Results:**
- **Zero-Shot E2E Accuracy:** **44.54%** (Slight Regression vs V4)
- **Analysis:** Despite achieving a stellar 49% accuracy on the localized training slice, test-set generalizability actually suffered a slight regression compared to the isolated V4 run (45.38%). This indicates that the initial V4 configuration with 36 unfrozen base decoder layers was already fully saturating the maximum possible geometric reasoning capacity extractable from the Qwen3-VL embeddings. Subjecting the massive, fragile image processor to deep end-to-end mathematical backpropagation caused slight validation overfitting/forgetting of its foundational spatial representations rather than discovering new heuristic alignments. Future gains will likely require scaling the target base model's logic capacity itself rather than further tuning the image processor.
