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

from models.latent_euclid import LatentEuclid
from training.stable_alignment_loss import AlignmentLossFactory
# from training.augmentation import GeometrySafeAugmentation # Temporarily isolated depending on vision model specifics

def parse_args():
    parser = argparse.ArgumentParser(description="LatentEuclid X-Encoder Full SFT Loop")
    parser.add_argument("--config", type=str, default="training/config.yaml",
                        help="Path to YAML training configuration")
    return parser.parse_args()

def setup_ddp():
    """Initializes Distributed Data Parallel setup."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    else:
        # Fallback to local CPU/Single GPU for testing if not launched via torchrun
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

class GeoThoughtsDataset(Dataset):
    """
    Parses the JSONL generation pairs, loads the raw vision images, 
    and aligns them with the offline continuous 4-step .pt manifolds.
    """
    def __init__(self, jsonl_path: str, targets_dir: str, tokenizer, k_steps=4):
        self.data = []
        self.targets_dir = targets_dir
        self.tokenizer = tokenizer
        self.k_steps = k_steps
        
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                
                # Verify that the target tensor actually exists
                target_path = os.path.join(targets_dir, f"problem_{idx}_targets.pt")
                if os.path.exists(target_path):
                    item["target_path"] = target_path
                    self.data.append(item)
                    
        print(f"Loaded {len(self.data)} valid aligned geometric datasets.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Image
        img_path = item["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Fallback mock image if path is broken
            image = Image.new('RGB', (224, 224), color = (73, 109, 137))
            
        # 2. Text Prompts (appended with K thoughts natively)
        # Note: the prompt needs the specialized <thought> sequences generated inside
        thought_string = "".join([f"<thought_{i+1}>" for i in range(self.k_steps)])
        full_text = item["question"] + " " + thought_string
        
        # 3. Target Manifolds pre-generated into .pt
        target_tensor = torch.load(item["target_path"], map_location="cpu", weights_only=True)
        
        return {
            "image": image,
            "text": full_text,
            "target": target_tensor # Shape [4, target_dim]
        }
        
def custom_collate(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])
    return images, texts, targets

def train():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    local_rank = setup_ddp()
    is_distributed = isinstance(local_rank, int)
    is_master = (local_rank == 0 if is_distributed else True)
    
    if is_master:
        wandb.init(
            project="LatentEuclid",
            name=f"LatentEuclid-{config['training']['loss_type']}",
            config=config
        )
    
    print(f"[{local_rank}] Instantiating LatentEuclid module constraints...")
    
    model = LatentEuclid(
        base_model_id=config["model"]["base_model_id"],
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    )
    
    if is_distributed:
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        model = model.to(local_rank) # cpu/cuda
        
    criterion = AlignmentLossFactory(loss_type=config["training"]["loss_type"])
    if is_distributed:
        criterion = criterion.to(local_rank)
        
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["training"]["learning_rate"]), 
        weight_decay=float(config["training"]["weight_decay"])
    )
    
    # Instantiate Data Loader
    dataset = GeoThoughtsDataset(
        jsonl_path=config["data"]["jsonl_path"],
        targets_dir=config["data"]["targets_dir"],
        tokenizer=model.module.tokenizer if is_distributed else model.tokenizer,
        k_steps=config["model"]["k_steps"]
    )
    
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=int(config["training"]["batch_size"]), 
        sampler=sampler,
        collate_fn=custom_collate,
        shuffle=(sampler is None)
    )
    
    print(f"[{local_rank}] Successfully initialized epoch pipelines!")
    
    epochs = int(config["training"]["epochs"])
    device = local_rank if is_distributed else local_rank
    
    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        
        for batch_idx, (images, texts, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Since VLMs (like Qwen3) usually require a complex processor for their images rather than raw PIL...
            # We extract them utilizing the associated model processor dynamically:
            tokenizer = model.module.tokenizer if is_distributed else model.tokenizer
            inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            
            # Note: For strict Qwen3-VL, you pass 'pixel_values' explicitly from its `Processor`. 
            # If `pixel_values` aren't defined, the model dynamically routes to text-only mode processing.
            predicted_latents = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
            targets = targets.to(device=device, dtype=predicted_latents.dtype)
            
            # Loss alignment mapping
            loss, metrics_dict = criterion(predicted_latents, targets)
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
            optimizer.step()
            
            if is_master and batch_idx % 2 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} | Batch {batch_idx} | {config['training']['loss_type']} Loss: {loss.item():.4f}")
                
                # Push tracked metrics to WandB securely
                metrics_dict["train/total_loss"] = loss.item()
                metrics_dict["train/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                metrics_dict["train/learning_rate"] = current_lr
                metrics_dict["epoch"] = epoch
                
                wandb.log(metrics_dict)
                
    if is_master:
        save_model = model.module if is_distributed else model
        torch.save(save_model.state_dict(), "latent_euclid_x_encoder_final.pt")
        print("Model state successfully saved.")
        wandb.finish()

if __name__ == "__main__":
    train()
