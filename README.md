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
