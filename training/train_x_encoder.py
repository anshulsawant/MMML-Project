import argparse
import yaml
import json
import os
import shutil
import time
import torch
import wandb
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from models.latent_euclid import LatentEuclid
from training.stable_alignment_loss import AlignmentLossFactory
from training.augmentation import GeometrySafeAugmentation

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
    def __init__(self, jsonl_path: str, targets_dir: str, tokenizer, k_steps=4, augment=False):
        self.data = []
        self.targets_dir = targets_dir
        self.tokenizer = tokenizer
        self.k_steps = k_steps
        self.augmentor = GeometrySafeAugmentation() if augment else None
        
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
            if self.augmentor is not None:
                image = self.augmentor(image)
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
        
    criterion = AlignmentLossFactory(
        loss_type=config["train_x_encoder"]["loss_type"],
        vicreg_sim_coeff=float(config["train_x_encoder"].get("vicreg_sim_coeff", 25.0)),
        vicreg_var_coeff=float(config["train_x_encoder"].get("vicreg_var_coeff", 25.0)),
        vicreg_cov_coeff=float(config["train_x_encoder"].get("vicreg_cov_coeff", 1.0))
    )
    if is_distributed:
        criterion = criterion.to(local_rank)
        
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["train_x_encoder"]["learning_rate"]), 
        weight_decay=float(config["train_x_encoder"]["weight_decay"])
    )
    
    # ---------------------------------------------------------
    # Checkpointing Logic (Load)
    # ---------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_dir = config["train_x_encoder"].get("checkpoint_dir", "/workspace/checkpoints")
    experiment_name = config.get("experiment", {}).get("name", "default")
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    latest_cp_path = None
    
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("x_encoder_epoch_") and f.endswith(".pt")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
            latest_cp_file = checkpoints[-1]
            latest_cp_path = os.path.join(checkpoint_dir, latest_cp_file)
            start_epoch = int(latest_cp_file.split('epoch_')[1].split('.')[0]) + 1
            
            print(f"[{local_rank}] Found checkpoint {latest_cp_file}. Resuming from epoch {start_epoch}...")
            
            # Load states
            cp = torch.load(latest_cp_path, map_location="cpu")
            if "val_loss" in cp:
                best_val_loss = cp.get("val_loss", float('inf'))
                
            if is_distributed:
                model.module.load_state_dict(cp["model_state_dict"])
            else:
                model.load_state_dict(cp["model_state_dict"])
            optimizer.load_state_dict(cp["optimizer_state_dict"])
            
            # Explicitly override the loaded learning rate with the config value
            # (optimizer.load_state_dict will otherwise overwrite the new lr with the old saved lr)
            target_lr = float(config["train_x_encoder"]["learning_rate"])
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr
            print(f"[{local_rank}] Checkpoint Optimizer LR manually overridden to {target_lr}")
    else:
        if is_master:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"[{local_rank}] Checkpoint directory {checkpoint_dir} created. Starting from scratch.")
    # ---------------------------------------------------------
    
    # Instantiate Data Loader
    full_dataset = GeoThoughtsDataset(
        jsonl_path=config["data"]["jsonl_path"],
        targets_dir=config["data"]["targets_dir"],
        tokenizer=model.module.tokenizer if is_distributed else model.tokenizer,
        k_steps=config["model"]["k_steps"],
        augment=True # Apply affine geometry-safe augmentations dynamically
    )
    
    # 90/10 Train/Val Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=int(config["train_x_encoder"]["batch_size"]), 
        sampler=train_sampler,
        collate_fn=custom_collate,
        shuffle=(train_sampler is None),
        drop_last=True
    )
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=int(config["train_x_encoder"]["batch_size"]), 
        sampler=val_sampler,
        collate_fn=custom_collate,
        shuffle=False
    )
    
    gradient_accumulation_steps = int(config["train_x_encoder"].get("gradient_accumulation_steps", 1))
    print(f"[{local_rank}] Successfully initialized epoch pipelines! Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    epochs = int(config["train_x_encoder"]["epochs"])
    max_steps_per_epoch = config["train_x_encoder"].get("max_steps_per_epoch", None)
    device = local_rank if is_distributed else local_rank
    
    for epoch in range(start_epoch, epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        avg_val_loss = float('inf')
        
        for batch_idx, (images, texts, targets) in enumerate(train_dataloader):
            if max_steps_per_epoch is not None and batch_idx >= max_steps_per_epoch:
                print(f"[{local_rank}] Reached max_steps_per_epoch ({max_steps_per_epoch}). Ending epoch {epoch} early.")
                break
            if batch_idx % gradient_accumulation_steps == 0:
                step_start_time = time.time()
                
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
            
            # Scale loss by accumulation steps
            # Dynamically handle the remainder of the epoch if the last accumulation step isn't full
            if (batch_idx + 1 == len(train_dataloader)) and (len(train_dataloader) % gradient_accumulation_steps != 0):
                current_accumulation_steps = len(train_dataloader) % gradient_accumulation_steps
            else:
                current_accumulation_steps = gradient_accumulation_steps
                
            loss = loss / current_accumulation_steps
            loss.backward()
            
            # Step conditionally based on batch_idx and accumulation steps
            if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train_x_encoder"]["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad()
                
                # --- RUN VALIDATION ON STEP ---
                model.eval()
                total_val_loss = 0.0
                total_val_mse = 0.0
                with torch.no_grad():
                    for val_idx, (val_img, val_txt, val_targ) in enumerate(val_dataloader):
                        val_inputs = tokenizer(val_txt, padding=True, return_tensors="pt").to(device)
                        val_pred = model(input_ids=val_inputs.input_ids, attention_mask=val_inputs.attention_mask)
                        val_targ = val_targ.to(device=device, dtype=val_pred.dtype)
                        
                        v_loss, v_metrics = criterion(val_pred, val_targ)
                        total_val_loss += v_loss.item()
                        if "loss/invariance_cos" in v_metrics:
                            total_val_mse += v_metrics["loss/invariance_cos"]
                        
                avg_val_loss = total_val_loss / max(1, len(val_dataloader))
                avg_val_mse = total_val_mse / max(1, len(val_dataloader))
                
                if is_master:
                    step_duration = time.time() - step_start_time
                    current_lr = optimizer.param_groups[0]['lr']
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    var_std_val = metrics_dict.get("loss/variance_std_physical", 0.0) if type(metrics_dict) is dict else 0.0
                    var_loss_val = metrics_dict.get("loss/variance_loss", 0.0) if type(metrics_dict) is dict else 0.0
                    train_mse_val = metrics_dict.get("loss/invariance_cos", 0.0) if type(metrics_dict) is dict else 0.0
                    print(f"Epoch {epoch} | Step {batch_idx + 1} | Time: {step_duration:.2f}s | Train Loss: {loss.item() * current_accumulation_steps:.4f} | Train Cos: {train_mse_val:.4f} | Grad Norm: {grad_norm_val:.2f} | Var: {var_std_val:.3f} | Val Loss: {avg_val_loss:.4f} | Val Cos: {avg_val_mse:.4f}")
                    
                    # Push tracked metrics to WandB securely
                    metrics_dict["train/total_loss"] = loss.item() * current_accumulation_steps
                    metrics_dict["train/grad_norm"] = grad_norm_val
                    metrics_dict["train/learning_rate"] = current_lr
                    metrics_dict["val/total_loss"] = avg_val_loss
                    metrics_dict["val/invariance_cos"] = avg_val_mse
                    metrics_dict["epoch"] = epoch
                    
                    wandb.log(metrics_dict)
                
                model.train() # Return to training mode
                # ------------------------------
                
            else:
                pass # Just accumulating gradients

        # --- SAVE CHECKPOINT PER EPOCH ---
        if is_master:
            cp_path = os.path.join(checkpoint_dir, f"x_encoder_epoch_{epoch}.pt")
            save_model = model.module if is_distributed else model
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'val_loss': best_val_loss
            }, cp_path)
            print(f"[{local_rank}] Saved checkpoint: {cp_path}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_cp_path = os.path.join(checkpoint_dir, "x_encoder_best.pt")
                shutil.copy2(cp_path, best_cp_path)
                print(f"[{local_rank}] New best validation loss {best_val_loss:.4f}! Saved {best_cp_path}")
            
            # Keep only latest 4 checkpoints to safely utilize the 500GB RunPod MooseFS disk quota
            old_epochs = [f for f in os.listdir(checkpoint_dir) if f.startswith("x_encoder_epoch_") and f.endswith(".pt")]
            if len(old_epochs) > 4:
                old_epochs.sort(key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
                for old_f in old_epochs[:-4]: # Keep the latest 4 checkpoints
                    try:
                        os.remove(os.path.join(checkpoint_dir, old_f))
                        print(f"[{local_rank}] Auto-deleted ancient checkpoint {old_f} to preserve disk space.")
                    except OSError:
                        pass
    if is_master:
        save_model = model.module if is_distributed else model
        torch.save(save_model.state_dict(), "latent_euclid_x_encoder_final.pt")
        print("Model state successfully saved.")
        wandb.finish()

if __name__ == "__main__":
    train()
