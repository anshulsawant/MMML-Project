# LatentEuclid & GeoThought: Enhancing Mathematical Geometry Reasoning in Vision-Language Models

This repository contains the official codebase for the **LatentEuclid** macro-JEPA geometry architecture and the **GeoThought** autoregressive baselines, built to solve the "Translation Bottleneck" and "Visual Forgetting" in state-of-the-art vision-language models (VLMs). 

Developed for the MMML 11-977 course project.

---

## 🏗️ Architecture Pathways

This repository is split into two primary architectures.

### 1. LatentEuclid (Macro-JEPA Continuous Alignment)
LatentEuclid completely bypasses the discrete autoregressive text bottleneck. Leveraging the Joint Embedding Predictive Architecture (VL-JEPA), LatentEuclid uses the **Qwen3-VL-4B-Instruct** encoder to map geometry inputs directly into a sequence of continuous frozen thought vectors predicting a mathematical manifold formulated by a smaller Expert Decoder (**Qwen3-0.6B**).
- Located internally in the `models/`, `training/`, `data/`, and `eval/` directories.

### 2. GeoThought (Autoregressive Baseline)
The initial baseline fine-tunes large vision-language models like **InternVL3-8B** and **Qwen2.5-VL** directly on the `GeoThought-6k` dataset. These models utilize standard autoregressive loops and CoT (Chain of Thought), often resulting in "Visual Forgetting" on longer proofs where language priors override diagram grounding. 
- Located internally in the `GeoThought/` and `Qwen25 LM/` directories.

---

## 🚀 Step-by-Step Guide: LatentEuclid Pipeline

### Phase 1: Environment Setup

To run LatentEuclid, initialize the virtual environment directly containing `torch`, `transformers`, `accelerate`, and API support.

```bash
cd MMML-Project/
python3 -m venv venv
source venv/bin/activate
pip install torch transformers accelerate python-dotenv openai scikit-learn torchvision google-genai
```

Add your Gemini API Key for data generation into an environment file (`~/.env`):
```env
GEMINI_API_KEY=AIzaSyB...
```

### Phase 2: Data Engineering (Geometry Extraction & Reasoning)

Our pipeline uses an Expert Teacher model (Gemini 3.1 Pro / Gemini 3 Flash Preview) to generate mathematically solid "K=4" step-by-step reasoning chains for geometric proofs. These text chains are then transformed into a continuous embedding manifold.

**Step 2a: Extract the Parquet Datasets**
First, unpack your raw geometric datasets from Huggingface Parquet files.
```bash
python -m data.extract_parquet --limit 6200
```

**Step 2b: Generate Reasoning Chains via Gemini Batch API**
We send thousands of problem images to Google's asynchronous Batch API to prevent running out of rate limits (429/503 errors) on synchronous requests.
```bash
# This packages all local problems and submits them securely to Google's backend.
python -m data.generate_geothoughts --limit 6243
```
If your script or internet gets interrupted while polling, you can seamlessly resume downloading the job outputs by providing the job URI:
```bash
python -m data.generate_geothoughts --resume_job batches/YOUR_JOB_ID_HERE
```
*Outputs are mapped safely onto physical newlines into `data/geothoughts_k4.jsonl`.*

**Step 2c: Data Verification & Ground Truth Extraction**
Because VLMs are highly verbose, simply generating the answers isn't enough for structural alignment. We must rigorously test Gemini's zero-shot generations against the dataset's native ground truths.
```bash
python -m data.evaluate_generated
```
* **Performance:** Gemini 3.1 Flash-Preview achieves a **95.16%** zero-shot accuracy evaluating geometric proofs.
* **Extraction Challenges:** Building this evaluation pipeline required robust logic to parse multimodal math. Ground truths often contain strict primitives (e.g. `10/3`), while VLMs emit varied formats (e.g., `3.3333`, `10\pi`, `sq units`, `\boxed{...}`). The `evaluate_generated.py` script utilizes complex LaTeX-stripping heuristics and a safe Python `ast.parse` / `eval` float-equivalence fallback structure to accurately cross-validate geometric answers despite trailing text and varying unit representations.

**Dataset Glossary:**
* `data/geothoughts_k4.jsonl`: Raw, unverified text outputs from the Gemini API.
* `data/ground_truths.json`: Raw target answers extracted from the original geometry datasets.
* `data/geothoughts_verified.jsonl`: The final, purely accurate subset of proofs where Gemini mathematically matched the ground truth safely. (Used for training the $X$-Encoder).

**Step 2d: Build the Target Manifold**
Pass the strictly formatted 4-step chains natively through the frozen `Qwen3-4B-Base` text model to create the `.pt` embedding vector targets.
```bash
python -m data.build_manifold
```

### Phase 3: CPU Smoke Testing (Optional but Recommended)

Since GPU compute is extremely expensive, you can rigorously test the entire structural LatentEuclid pipeline (forward passes, loss calculations, index checking, yaml parsing, backward gradients) using the microscopic `trl-internal-testing/tiny-Qwen3VL` models natively on your CPU. This executes in under two seconds.
```bash
python -m tests.test_architecture
python -m tests.test_data_pipeline
```

### Phase 4: Training (Continuous Alignment)

Training relies heavily on `training/config.yaml` to configure data paths and learning rates, and uses a dynamic loss factory natively implementing **Vanilla InfoNCE**, **Thresholded InfoNCE**, and **VICReg**.

```bash
# Example training run using VICReg (Variance-Invariance-Covariance Regularization)
python -m training.train_x_encoder \
    --model_id "Qwen/Qwen3-VL-4B-Instruct" \
    --loss_type "vicreg" \
    --batch_size 4
```

*Note: For `vicreg`, our pipeline natively utilizes structurally safe geometry augmentations (Affine transformations like shearing, rotation, scaling) and avoids destructive random cropping via `training/augmentation.py`.*

### Phase 4.5: Y-Decoder Prefix Projection (Training)

Once the X-Encoder is heavily trained to map images to geometry, we execute Phase 4.5 to link the continuous topology into discretely generated targets using a new Base LLM (e.g., `Qwen3-4B-Base`). This teaches a 2-layer MLP to map the frozen $X$-Encoder geometric thoughts into the structural embedding layer of the $Y$-Decoder, allowing standard autoregressive mathematical generation.

```bash
python -m training.train_decoder_projection \
    --config training/config_decoder.yaml \
    --x_encoder_weights checkpoints/x_encoder_best.pt
```

For experimental End-to-End alignment unfreezing the X-Encoder limits, append `--end_to_end`.

### Phase 5: Generative Inference Validation

To test whether the Target Decoder has successfully learned how to unwrap the geometric logic from the continuous Prefix Tensors, run the generative suite. This computes purely offline predictions of the test split numbers formatted to explicitly cue the `<|im_end|>` termination tokens.

```bash
python -m validate_generation
```

### Phase 6: Probing & Structural Evaluation

The evaluation suite contains scripts measuring exact time-to-answer latency reductions. It runs the primary "Visual Forgetting" thesis probe, calculating cosine similarity and Euclidean drift of the `LatentEuclid` predicted thought vectors against baseline generative representations across deep layers.

```bash
python -m eval.temporal_probing
```

---

## 🛠️ Step-by-Step Guide: GeoThought (Baselines)

For comparative benchmarking or utilizing our pre-trained autoregressive checkpoints:

### Resources
- **Datasets**: 
  - [GeoThought-6k](https://huggingface.co/datasets/xinlingdedeng/GeoThought-6k)
  - [Geo-Thought-Augmented-10K](https://huggingface.co/datasets/xinlingdedeng/Geo-Thought)
- **Baseline Models**: [InternVL3-8B-10834](https://huggingface.co/xinlingdedeng/InternVL3-8B-10834)

### Evaluation Instructions
```bash
cd GeoThought
conda create -n geothought python=3.10 -y
conda activate geothought
pip install -e .

# Run Evaluation Bash Pipeline
bash scripts/eval_multi.sh \
    path-to-model \
    playground/data/test_questions.jsonl \
    path-to-output \
    path-to-image-folder \
    num_gpus \
    temperature
```

---

## 🤝 Acknowledgement
This project builds upon previous work in structured geometric reasoning and multimodal representation fissions. We thank the research community for their foundational contributions to geometry benchmarks and the ARC-AGI problem domains.

---

## 🕒 Complete Commit History
- **adf4c71** docs: Add ground truth extraction challenges, dataset glossary, and Gemini zero-shot performance to README (Anshul Sawant)
- **e82a3fe** docs: Update LatentEuclid pipeline to reflect base model prefix integration and validation instructions (Anshul Sawant)
- **b63f0eee** Pass eos_token_id properly to generation output (Anshul Sawant)
- **3a673f7** Pass eos_token_id properly to generation output (Anshul Sawant)
- **7aefebbe** Reduce learning rate and add Answer delimiter to prompt to cue Qwen base model (Anshul Sawant)
- **371a4bf** Reduce learning rate and add Answer delimiter to prompt to cue Qwen base model (Anshul Sawant)
- **06ae8ce** Remove thought removal to verify pure zero shot capability of base models without syntax leakage (Anshul Sawant)
- **184d09f** Add Qwen3-4B base target and explicitly bypass empty string hallucination prompt bugs (Anshul Sawant)
- **abcedf0** Fix token padding loop leakage in CE loss calculation and fix text/geom concatenation alignment (Anshul Sawant)
- **66a7b73** Fix GeoThought gitignore path blocking multimodal image reads (Anshul Sawant)
- **a16fedc** docs: Add LatentEuclid whitepaper artifact formalizing VL-JEPA methodology (Anshul Sawant)
- **e1973e1** docs: Completely expand step-by-step documentation for LatentEuclid architecture, data engine, and training lifecycle (Anshul Sawant)
- **4c597b9** docs: Append recent project commit history to README (Anshul Sawant)
- **b2d426d** fix: Patch json newline rendering bug in extraction array and add resume_job checkpointing capability (Anshul Sawant)
- **7876ec4** Merge branch 'main' of https://github.com/anshulsawant/MMML-Project (SMa2021)
- **f59b749** simple multimodal model (SMa2021)
- **ca5a7ab** refactor: Migrate generator to Gemini Batch API targeting flash-preview to resolve synchronous quota exhaustion (Anshul Sawant)
- **9358ef0** fix: Resolve AsyncOpenAPI strict token cutoff mapping for 4-step generation blocks (Anshul Sawant)
- **0281003** data: Add initial batch of generated K=4 reasoning chains for geothought dataset (Anshul Sawant)
- **e5e495a** fix: Strengthen K=4 system prompt to explicitly prevent model truncations (Anshul Sawant)
- **e2f4aac** feat: Add dynamic limits via argparse to data pipeline generators (Anshul Sawant)
- **f5afa43** feat: Remove batch count hard caps for generation pipeline and include token tracking logic (Anshul Sawant)
- **13015d6** test: Implement full e2e end-to-end training smoke test and fix Target bfloat16 cast mapping (Anshul Sawant)
- **9cdcb86** test: Add smoke tests for SFT training pipeline data mapping and yaml parses (Anshul Sawant)
- **ea7488e** feat: Implement formal data plumbing and training loop yaml configs for SFT (Anshul Sawant)
- **745405f** feat: Implement batch inference for target manifold generation (Anshul Sawant)
- **bb0dcae** update: Target gemini-3.1-pro-preview endpoint (Anshul Sawant)
- **ee49b69** fix: Unwrap parquet binary correctly to serve into generate_geothoughts via local jsonl test structure (Anshul Sawant)
- **861bcbd** feat: Integrate GeoThought local dataset into Gemini generator pipeline (Anshul Sawant)
- **56b908b** test: Add smoke tests for data engineering and manifold generation (Anshul Sawant)
- **5143322** feat: Add LatentEuclid architecture, training, data gen, and eval scripts (Anshul Sawant)
- **a897d1e** A simple util to generate stats from results json. (Anshul Sawant)
- **bcfa4ab** add thinking with sample examples for molmo2 (SMa2021)
- **855762d** Add sampling (SMa2021)
- **9266924** Fix typo (Joanna Smolska)
- **b4b7737** Add Qwen 25 Laguage Only (Joanna Smolska)
- **671e701** add colab (SMa2021)
- **dc3cbfb** Merge branch 'main' into Qwen3-Baseline (stl008)
- **36219ed** Merge branch 'main' of https://github.com/anshulsawant/MMML-Project (SMa2021)
- **e6a877a** add more results (SMa2021)
- **9241873** Generate missing evaluation stats for base model (Anshul Sawant)
- **23fadbc** Add benchmark results from base InternVL3-8B-Instruct model (Anshul Sawant)
- **2afc5ef** Add custom chat template to fix InternVL3 concatenation bug and restore missing vision flags (Anshul Sawant)
- **b2780be** Allow HuggingFace Hub IDs as MODEL_PATH by conditionally applying realpath (Anshul Sawant)
- **3aceb1b** Merge branch 'main' of https://github.com/anshulsawant/MMML-Project (SMa2021)
- **9e7a274** other tests (SMa2021)
- **13fd537** monet benchmark (Canchen Li)
- **bd08c96** Fix inference script path bug and multimodal payload structure (Anshul Sawant)
- **c95f07a** Finetuned Intern3-8B model results. (Anshul Sawant)
- **1a5db2c** Add check for valid realpath model directory to prevent empty vllm --model argument (Anshul Sawant)
- **67d4fbd** Use realpath for vLLM local model directory to prevent huggingface parsing errors (Anshul Sawant)
- **11efe89** Make all shell scripts executable (Anshul Sawant)
- **498263d** Translate all Chinese comments to English (Anshul Sawant)
- **45a6628** Translate inference script output to English (Anshul Sawant)
- **d38f717** Increase max_workers to 32 for higher inference throughput (Anshul Sawant)
- **c5f5f20** Enhance run_benchmark_remote.sh to exit early on vLLM server crash (Anshul Sawant)
- **ebeed6d** Add benchmark scripts and convert GeoThought to standard directory (Anshul Sawant)
- **a2bc243** added Qwen3-VL baselines + processed dataset (stl008)
- **1af197a** add analytics (SMa2021)
- **e6b3703** results (SMa2021)
- **c9061a7** molmo2 full results (SMa2021)
- **1fd7bcd** molmo2 v1 (SMa2021)
- **d79f772** Upload of report and GeoThought. (Anshul Sawant)
