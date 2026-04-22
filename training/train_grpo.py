import os
import torch
import json
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from models.latent_euclid import LatentEuclid
from models.y_decoder_prefix import YDecoderPrefix
from models.grpo_euclid_wrapper import UnifiedEuclidGRPO
from training.reward_functions import accuracy_reward_func, format_reward_func

def load_geothoughts_dataset(json_path="data/geothoughts_arbitrary_cot.jsonl"):
    """
    Format our existing variable length target dataset 
    into TRL's expected conversational structures mapping prompts cleanly.
    """
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # TRL handles list of conversational turns
            # For VLM alignment, we pack images inside standard contexts!
            prompt = item["question"]
            
            # Pack dynamically
            data.append({
                "prompt": [{"role": "user", "content": prompt}], 
                # Meta contexts safely caching targets
                "image_path": item["image_path"],
                "target_math": item.get("target_math", "")
            })
            
    return Dataset.from_list(data)

import yaml

def main():
    print("Initializing Unsharded GRPO Target Sequence Protocol...")
    with open("training/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    grpo_cfg = config.get("train_grpo", {})
    
    # 1. Instantiate the singular networks tracking parameters actively!
    base_encoder = LatentEuclid(
        base_model_id=config["model"].get("base_model_id", "Qwen/Qwen3-VL-4B-Instruct"),
        target_model_id=config["model"].get("target_model_id", "Qwen/Qwen3-VL-4B-Instruct"),
        max_thought_tokens=config["model"].get("max_thought_tokens", 30)
    )
    
    enc_weights = grpo_cfg.get("x_encoder_weights_override")
    if enc_weights and os.path.exists(enc_weights):
        state_dict = torch.load(enc_weights, map_location="cpu")
        base_encoder.load_state_dict(state_dict.get("model_state_dict", state_dict))
        print(f"Loaded SFT X-Encoder weights from {enc_weights}")
        
    base_decoder = YDecoderPrefix(
        target_model_id=config["model"].get("target_model_id", "Qwen/Qwen3-VL-4B-Instruct"),
        unfreeze_layers=-1,
        use_projection_mlp=config.get("train_grpo", {}).get("use_projection_mlp", True)
    )
    
    dec_weights = grpo_cfg.get("decoder_weights_override")
    if dec_weights and os.path.exists(dec_weights):
        state_dict = torch.load(dec_weights, map_location="cpu")
        base_decoder.load_state_dict(state_dict.get("model_state_dict", state_dict))
        print(f"Loaded SFT Y-Decoder weights from {dec_weights}")
    
    # 2. Wrap mathematically
    from models.grpo_euclid_wrapper import EuclidGRPOConfig
    grpo_config = EuclidGRPOConfig()
    grpo_config.hidden_size = base_encoder.config.hidden_size
    model = UnifiedEuclidGRPO(config=grpo_config, x_encoder=base_encoder, y_decoder=base_decoder)
    
    dataset = load_geothoughts_dataset()
    tokenizer = base_decoder.tokenizer
    
    # 3. Configure TRL GRPO Training loop scaling cleanly to the H200
    # Because H200 packs 141 GB of VRAM, 8B params fit safely globally scaling parallel rollout 
    # sequences dynamically without complex multi-node topology dependencies.
    training_args = GRPOConfig(
        output_dir=f"checkpoints/{grpo_cfg.get('experiment_name', 'grpo_euclid')}",
        learning_rate=float(grpo_cfg.get("learning_rate", 1e-5)),
        per_device_train_batch_size=int(grpo_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(grpo_cfg.get("gradient_accumulation_steps", 4)),
        max_completion_length=int(grpo_cfg.get("max_completion_length", 64)),
        num_generations=int(grpo_cfg.get("num_generations", 4)),
        bf16=True,
        remove_unused_columns=False,
        log_level="info",
        report_to="wandb"
    )
    
    print("Initiating GRPOTrainer scaling parallel optimization over the 8B target...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[accuracy_reward_func, format_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
