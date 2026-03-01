# LatentEuclid & GeoThought: Enhancing Mathematical Geometry Reasoning in Vision-Language Models

This repository contains the official codebase for the **LatentEuclid** macro-JEPA geometry architecture and the **GeoThought** autoregressive baselines, built to solve the "Translation Bottleneck" and "Visual Forgetting" in state-of-the-art vision-language models (VLMs). 

Developed for the MMML 11-977 course project.

---

## 🏗️ Architecture Pathways

This repository is split into two primary architectures.

### 1. GeoThought (Autoregressive Baseline)
The initial baseline fine-tunes large vision-language models like **InternVL3-8B** and **Qwen2.5-VL** directly on the `GeoThought-6k` dataset. These models utilize standard autoregressive loops and CoT (Chain of Thought), often resulting in "Visual Forgetting" on longer proofs where language priors override diagram grounding. 
- Located internally in the `GeoThought/` and `Qwen25 LM/` directories.

### 2. LatentEuclid (Macro-JEPA Continuous Alignment)
LatentEuclid completely bypasses the discrete autoregressive text bottleneck. Leveraging the Joint Embedding Predictive Architecture (VL-JEPA), LatentEuclid uses the **Qwen3-VL-4B-Instruct** encoder to map geometry inputs directly into a sequence of continuous frozen thought vectors predicting a mathematical manifold formulated by a smaller Expert Decoder (**Qwen3-0.6B**).
- Located internally in the `models/`, `training/`, `data/`, and `eval/` directories.

---

## 🚀 Quick Start: LatentEuclid

### 1. Environment Setup

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

### 2. Data Engineering & Manifold Generation
Using Gemini 3.1 Pro as an "Expert Teacher", we prompt-engineer structured 4-step geometry reasoning chains, then translate those chains into continuous `target_dim=1024` vectors mapping to the Qwen3-0.6B internal dimensions.

```bash
# Query the Gemini 3.1 Pro API for K=4 parsing
python -m data.generate_geothoughts

# Embed the generated steps natively into continuous .pt tensors
python -m data.build_manifold
```

### 3. CPU Smoke Testing
Since GPU compute is expensive, you can rigorously test the entire structural LatentEuclid pipeline (forward passes, loss calculations, index checking, backward gradients) using the absolute microscopic `trl-internal-testing/tiny-Qwen3VL` models directly on your CPU. This executes in under a second:

```bash
python -m tests.test_architecture
```

### 4. Training (Continuous Alignment)
Training uses a distributed DDP wrapper mapping massive Qwen3-VL visual outputs onto the Target Manifold. We support a dynamic loss factory natively implementing **Vanilla InfoNCE**, **Thresholded InfoNCE**, and **VICReg**.

```bash
python -m training.train_x_encoder --model_id "Qwen/Qwen3-VL-4B-Instruct" --loss_type "vicreg" --batch_size 4
```

*Note: For `vicreg`, our pipeline natively utilizes structurally safe geometry augmentations (Affine transformations like shearing and scaling) and strictly avoids destructive cropping inside `training/augmentation.py`.*

### 5. Probing & Inference Eval
The evaluation suite contains scripts measuring exact time-to-answer latency reductions and the primary "Visual Forgetting" thesis probe comparing `LatentEuclid` predicted thought vectors against the `baseline` pre-generation hidden states.

```bash
python -m eval.temporal_probing
```

---

## 🛠️ Quick Start: GeoThought (Baselines)

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

## 🕒 Recent Commit History
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
- **36219ed** Merge branch 'main' of https://github.com/anshulsawant/MMML-Project (SMa2021)
