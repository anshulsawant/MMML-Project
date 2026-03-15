# Experiment Logs

## v2_huber_mean_pooled
- **Objective:** Train LatentEuclid X-Encoder to map Vision tokens to fixed LLM manifold.
- **Micro-Batch Size:** 32 (Gradient Checkpointing Enabled)
- **Gradient Accumulation:** 4 (Global Batch Size: 128)
- **Loss:** Huber-Cosine (Cosine weighted 10x)
- **Throughput:** ~75s per global step on A100.
