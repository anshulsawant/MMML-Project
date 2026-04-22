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

def main():
    print("Initializing Unsharded GRPO Target Sequence Protocol...")
    
    # 1. Instantiate the singular networks tracking parameters actively!
    base_encoder = LatentEuclid(max_thought_tokens=30)
    base_decoder = YDecoderPrefix(unfreeze_layers=-1)
    
    # 2. Wrap mathematically
    model = UnifiedEuclidGRPO(config=base_encoder.config, x_encoder=base_encoder, y_decoder=base_decoder)
    
    dataset = load_geothoughts_dataset()
    tokenizer = base_decoder.tokenizer
    
    # 3. Configure TRL GRPO Training loop scaling cleanly to the H200
    # Because H200 packs 141 GB of VRAM, 8B params fit safely globally scaling parallel rollout 
    # sequences dynamically without complex multi-node topology dependencies.
    training_args = GRPOConfig(
        output_dir="checkpoints/grpo_euclid",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_prompt_length=256,
        max_completion_length=64, # Maximum length of reasoning text output generated!
        num_generations=4, # Rollout size per prompt optimized securely internally!
        bf16=True,
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
