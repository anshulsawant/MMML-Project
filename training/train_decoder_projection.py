import argparse
import yaml
import json
import os
import torch
import wandb
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import random

from models.latent_euclid import LatentEuclid
from models.y_decoder_prefix import YDecoderPrefix

def parse_args():
    parser = argparse.ArgumentParser(description="LatentEuclid Phase 4.5: Train Prefix Projection")
    parser.add_argument("--config", type=str, default="training/config_decoder.yaml",
                        help="Path to YAML training configuration")
    parser.add_argument("--x_encoder_weights", type=str, default="checkpoints/x_encoder_best.pt",
                        help="Path to the frozen VICReg-aligned X-Encoder weights")
    parser.add_argument("--end_to_end", action="store_true",
                        help="Experimental: Unfreeze X-Encoder to train end-to-end (Method B)")
    return parser.parse_args()

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"

class GeoThoughtsTextDataset(Dataset):
    """
    Extends the dataset to expose target answer texts for Perplexity Loss.
    """
    def __init__(self, data_list, ground_truths, tokenizer, k_steps=4):
        self.data = data_list
        self.ground_truths = ground_truths
        self.tokenizer = tokenizer
        self.k_steps = k_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Image
        img_path = item["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), color = (73, 109, 137))
            
        # 2. Base VLM Input Text
        thought_string = "".join([f"<thought_{i+1}>" for i in range(self.k_steps)])
        full_text = item["question"] + " " + thought_string
        
        # 3. Final Target Answer for the Y-Decoder to learn to generate
        # Lookup the true answer string dynamically from ground_truths.json
        target_answer = str(self.ground_truths.get(img_path, "0"))
        target_answer += "<|im_end|>" # Ensure it learns to stop

        return {
            "image": image,
            "text": full_text,
            "target_answer": target_answer
        }

def custom_collate(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    answers = [item["target_answer"] for item in batch]
    return images, texts, answers

def train():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    local_rank = setup_ddp()
    is_distributed = isinstance(local_rank, int)
    is_master = (local_rank == 0 if is_distributed else True)
    
    device = local_rank if is_distributed else local_rank

    if is_master:
        wandb.init(
            project="LatentEuclid",
            name=f"DecoderProjection-E2E-{args.end_to_end}",
            config=config
        )

    print(f"[{local_rank}] Loading Phase 3 LatentEuclid X-Encoder...")
    x_encoder = LatentEuclid(
        base_model_id=config["model"]["base_model_id"],
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    )
    
    # Load VICReg weights
    weight_path = config["model"].get("x_encoder_weights", args.x_encoder_weights)
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        x_encoder.load_state_dict(state_dict["model_state_dict"])
        print(f"[{local_rank}] Successfully loaded pre-aligned geometry from {weight_path}")
    else:
        print(f"Error: Could not find '{weight_path}'. You must complete Phase 3 VICReg first!")
        exit(1)

    if not args.end_to_end:
        print(f"[{local_rank}] Freezing X-Encoder (Method A)...")
        for param in x_encoder.parameters():
            param.requires_grad = False
        x_encoder.eval() # Ensure dropout/BN is frozen
    else:
        print(f"[{local_rank}] Unfreezing X-Encoder for End-to-End Co-Training (Method B)...")
        x_encoder.train()
        
    x_encoder = x_encoder.to(local_rank)

    print(f"[{local_rank}] Loading Phase 4 Y-Decoder Prefix...")
    y_decoder = YDecoderPrefix(
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    )
    y_decoder = y_decoder.to(local_rank)
    
    # y_decoder base LLM is always frozen in `__init__`.
    # `prefix_projection` intrinsically has requires_grad=True
    
    if is_distributed:
        if args.end_to_end:
            x_encoder = DDP(x_encoder, device_ids=[local_rank])
        y_decoder = DDP(y_decoder, device_ids=[local_rank])

    # Filter trainable parameters
    trainable_params = list(y_decoder.parameters())
    if args.end_to_end:
        trainable_params += list(x_encoder.parameters())

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainable_params), 
        lr=float(config["training"]["learning_rate"]), 
        weight_decay=float(config["training"]["weight_decay"])
    )

    # ------------------ Data Split ------------------
    with open(config["data"]["jsonl_path"], 'r') as f:
        full_data = [json.loads(line) for line in f]
        
    with open("data/ground_truths.json", 'r') as f:
        ground_truths = json.load(f)
        
    # 90/10 Train/Val Split for overfitting protection without sklearn
    random.seed(42)
    random.shuffle(full_data)
    split_idx = int(0.9 * len(full_data))
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    if is_master:
        print(f"Train split: {len(train_data)} | Val split: {len(val_data)}")

    x_tokenizer = x_encoder.module.tokenizer if is_distributed else x_encoder.tokenizer

    train_dataset = GeoThoughtsTextDataset(train_data, ground_truths, x_tokenizer, config["model"]["k_steps"])
    val_dataset = GeoThoughtsTextDataset(val_data, ground_truths, x_tokenizer, config["model"]["k_steps"])

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=int(config["training"]["batch_size"]), sampler=train_sampler, collate_fn=custom_collate, shuffle=(train_sampler is None))
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    val_loader = DataLoader(val_dataset, batch_size=int(config["training"]["batch_size"]), sampler=val_sampler, collate_fn=custom_collate, shuffle=False)

    epochs = int(config["training"]["epochs"])
    
    for epoch in range(epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        y_decoder.train() # Ensures projection layer tracks gradients
        if args.end_to_end:
            x_encoder.train()
            
        total_train_loss = 0.0
        
        for batch_idx, (images, texts, target_answers) in enumerate(train_loader):
            optimizer.zero_grad()
            
            inputs = x_tokenizer(texts, padding=True, return_tensors="pt").to(device)
            
            # Remove the <thought> placeholders from the string so the pure question is fed to the downstream LLM
            # The geometric latents will be concatenated seamlessly at the end
            decoder_prompts = []
            for t in texts:
                clean_t = t
                for i in range(config["model"]["k_steps"]):
                    clean_t = clean_t.replace(f"<thought_{i+1}>", "")
                decoder_prompts.append(clean_t.strip())
            
            # If Method A: X-Encoder is frozen, no gradients track through it
            if not args.end_to_end:
                with torch.no_grad():
                    # predicted_latents shape: [batch, K, hidden_dim]
                    predicted_latents = x_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            else:
                # Method B: Gradients track all the way back to the VLM image processor
                predicted_latents = x_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
            # Forward pass through Decoder (computes Cross Entropy against `target_answers`)
            outputs = y_decoder(
                predicted_latents=predicted_latents, 
                text_prompts=decoder_prompts,
                labels=target_answers
            )
            
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(trainable_params, float(config["training"]["max_grad_norm"]))
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if is_master and batch_idx % 2 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | CE Training Loss: {loss.item():.4f}")
                wandb.log({"train/ce_loss": loss.item(), "epoch": epoch})
                
        # ------------------ Validation Loop ------------------
        y_decoder.eval()
        if args.end_to_end:
            x_encoder.eval()
            
        total_val_loss = 0.0
        with torch.no_grad():
            for val_idx, (images, texts, target_answers) in enumerate(val_loader):
                inputs = x_tokenizer(texts, padding=True, return_tensors="pt").to(device)
                predicted_latents = x_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                
                val_prompts = []
                for t in texts:
                    clean_t = t
                    for i in range(config["model"]["k_steps"]):
                        clean_t = clean_t.replace(f"<thought_{i+1}>", "")
                    val_prompts.append(clean_t.strip() + " ")
                
                outputs = y_decoder(
                    predicted_latents=predicted_latents, 
                    text_prompts=val_prompts,
                    labels=target_answers
                )
                total_val_loss += outputs.loss.item()
                
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        if is_master:
            print(f"=== Epoch {epoch} Validation CE Loss: {avg_val_loss:.4f} ===")
            wandb.log({"val/ce_loss": avg_val_loss, "epoch": epoch})
            
            # Save epoch checkpoint
            checkpoint_dir = config["training"].get("checkpoint_dir", "/workspace/checkpoints/decoder")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            save_decoder = y_decoder.module if is_distributed else y_decoder
            checkpoint_path = os.path.join(checkpoint_dir, f"decoder_epoch_{epoch}.pt")
            torch.save(save_decoder.prefix_projection.state_dict(), checkpoint_path)
            print(f"[cuda] Saved checkpoint: {checkpoint_path}")
            
            if args.end_to_end:
                save_encoder = x_encoder.module if is_distributed else x_encoder
                encoder_path = os.path.join(checkpoint_dir, f"x_encoder_e2e_epoch_{epoch}.pt")
                torch.save(save_encoder.state_dict(), encoder_path)

    if is_master:
        print("Training successfully finished!")
        wandb.finish()

if __name__ == "__main__":
    train()
