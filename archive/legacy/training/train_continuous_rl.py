import os
import torch
import torch.nn as nn
import json
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from models.latent_euclid import LatentEuclid
from models.y_decoder_prefix import YDecoderPrefix
from training.train_grpo import load_geothoughts_dataset

def main():
    print("Initiating Continuous Latent REINFORCE Optimization Pipeline natively...")
    with open("training/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    grpo_cfg = config.get("train_grpo", {})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate Core Networks
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
        target_model_id=config["model"].get("decoder_base_model_id", "Qwen/Qwen3-4B-Base"),
        unfreeze_layers=-1,
        use_projection_mlp=grpo_cfg.get("use_projection_mlp", True)
    )
    
    dec_weights = grpo_cfg.get("decoder_weights_override")
    if dec_weights and os.path.exists(dec_weights):
        state_dict = torch.load(dec_weights, map_location="cpu")
        base_decoder.load_state_dict(state_dict.get("model_state_dict", state_dict))
        print(f"Loaded SFT Y-Decoder weights from {dec_weights}")
    
    # 2. Freeze Decoder structurally (we only train Encoder latents using RL)
    base_decoder.eval()
    for param in base_decoder.parameters():
        param.requires_grad = False
        
    # Unfreeze Encoder iteratively enabling Continuous Backpropagation
    base_encoder.train()
    for param in base_encoder.parameters():
        param.requires_grad = True

    base_encoder = base_encoder.to(device)
    base_decoder = base_decoder.to(device)
    
    # 3. Setup Dataset
    dataset = load_geothoughts_dataset()
    
    def collate_fn(batch):
        return batch[0] # batch_size=1 natively

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(base_encoder.parameters(), lr=float(grpo_cfg.get("learning_rate", 1e-5)))
    
    G = int(grpo_cfg.get("num_generations", 4))
    sigma = float(grpo_cfg.get("latent_sigma", 0.1)) # Variance constraint
    grad_accum_steps = int(grpo_cfg.get("gradient_accumulation_steps", 4))
    max_epochs = config.get("train_end_to_end", {}).get("epochs", 1)
    
    # Validation Reward Module
    from training.reward_functions import accuracy_reward_func

    # 4. Continuous REINFORCE Truncated Loop
    print(f"\n--- Training Details ---")
    print(f"Rollouts per sequence : {G}")
    print(f"Continuous Variance (Sigma) : {sigma}")
    print(f"Batch Size (Accumulation) : {grad_accum_steps}")
    print(f"------------------------\n")
    
    step_count = 0
    optimizer.zero_grad()
    
    for epoch in range(max_epochs):
        for data in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            
            prompt = data["prompt"] # Clean string
            img_path = data["image_path"]
            target_math = data["target_math"]
            
            # --- VISION BINDINGS ---
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            rendered = base_encoder.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = base_encoder.processor(text=[rendered], images=[img], padding=True, return_tensors="pt").to(device)
            
            # Extract latent mu
            dynamic_input_ids = inputs.input_ids
            dynamic_attention_mask = inputs.attention_mask
            dynamic_mm = inputs.get("mm_token_type_ids")
            pixel_vals = inputs.get("pixel_values")
            image_grid = inputs.get("image_grid_thw")
            
            max_steps = 4
            for step in range(max_steps):
                t_id = torch.tensor([[base_encoder.thought_ids[step]]], device=device)
                dynamic_input_ids = torch.cat([dynamic_input_ids, t_id], dim=1)
                new_attn = torch.ones((1, 1), dtype=dynamic_attention_mask.dtype, device=device)
                dynamic_attention_mask = torch.cat([dynamic_attention_mask, new_attn], dim=1)
                if dynamic_mm is not None:
                    new_mm_token = torch.zeros((1, 1), dtype=dynamic_mm.dtype, device=device)
                    dynamic_mm = torch.cat([dynamic_mm, new_mm_token], dim=1)
            
            predicted_latents_mu = base_encoder(
                input_ids=dynamic_input_ids, 
                pixel_values=pixel_vals, 
                attention_mask=dynamic_attention_mask,
                image_grid_thw=image_grid,
                mm_token_type_ids=dynamic_mm
            )
            
            # Computed Latents (Old Policy)
            mu_old = predicted_latents_mu.detach()
            dist_old = torch.distributions.Normal(mu_old.repeat(G, 1, 1), sigma)
            sampled_latents = dist_old.sample() # [G, K, D]
            sampled_latents = sampled_latents.detach()
            
            log_probs_old = dist_old.log_prob(sampled_latents).sum(dim=(1, 2)).detach() # [G]
            
            # --- EVALUATION DECODING ---
            # (Executed strictly once per batch)
            with torch.no_grad():
                gen_prompts = [prompt] * G
                max_new = int(grpo_cfg.get("max_completion_length", 15))
                outputs = base_decoder.generate(sampled_latents, text_prompts=gen_prompts, max_new_tokens=max_new)
                
            rewards = accuracy_reward_func(gen_prompts, outputs, [target_math]*G)
            rewards = torch.tensor(rewards, device=device, dtype=torch.bfloat16) # [G]
            
            if rewards.sum().item() == 0:
                adv = torch.zeros_like(rewards)
            else:
                adv = rewards - rewards.mean()
                # Optional GRPO Advantage normalization for structural stability:
                std = adv.std() if adv.std() > 0 else 1.0
                adv = adv / (std + 1e-4)
                
            # --- PPO INNER OPTIMIZATION EPOCHS ---
            # SoTA RL relies on iterative clipped updates to prevent topological catastrophic collapse
            ppo_epochs = int(grpo_cfg.get("ppo_epochs", 3))
            epsilon = float(grpo_cfg.get("ppo_epsilon", 0.2))
            
            for ppo_step in range(ppo_epochs):
                # Re-compute current policy mathematically against explicit computational graph
                predicted_latents_mu_current = base_encoder(
                    input_ids=dynamic_input_ids, 
                    pixel_values=pixel_vals, 
                    attention_mask=dynamic_attention_mask,
                    image_grid_thw=image_grid,
                    mm_token_type_ids=dynamic_mm
                )
                
                dist_current = torch.distributions.Normal(predicted_latents_mu_current.repeat(G, 1, 1), sigma)
                log_probs_current = dist_current.log_prob(sampled_latents).sum(dim=(1, 2)) # [G]
                
                # Importance Sampling Ratio natively bounding probabilities seamlessly
                ratio = torch.exp(log_probs_current - log_probs_old)
                
                # Structurally Clamped PPO Objective maximizing expectations safely!
                pg_loss1 = adv * ratio
                pg_loss2 = adv * torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
                loss = -torch.min(pg_loss1, pg_loss2).mean() / grad_accum_steps
                
                loss.backward()
                
                # Accumulation triggers updates mathematically identically inside the PPO constraints
                if (ppo_step + 1) == ppo_epochs:
                    step_count += 1
                    if step_count % grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(base_encoder.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        print(f"Step {step_count // grad_accum_steps} | PPO Loss: {loss.item()*grad_accum_steps:.4f} | R: {rewards.mean().item():.2f}")

    # Export Weights Iteration Safely
    out_dir = f"checkpoints/{grpo_cfg.get('experiment_name', 'continuous_latent_rl')}/x_encoder"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(base_encoder.state_dict(), f"{out_dir}/encoder_rl_best.pt")
    print("Optimization Completed. Weights locked.")

if __name__ == "__main__":
    main()
