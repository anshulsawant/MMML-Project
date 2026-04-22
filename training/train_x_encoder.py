import argparse
import yaml
import json
import os
import shutil
import subprocess
import threading
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

# ---------------------------------------------------------------------------
# RunPod S3 + HuggingFace Hub background upload helpers
# ---------------------------------------------------------------------------

RUNPOD_S3_REGION = "us-md-1"
RUNPOD_S3_ENDPOINT = "https://s3api-us-md-1.runpod.io"

_upload_threads: list[threading.Thread] = []


def _s3_base_args() -> list[str]:
    return ["--region", RUNPOD_S3_REGION, "--endpoint-url", RUNPOD_S3_ENDPOINT]


def _s3_upload_file(local_path: str, bucket: str, s3_key: str) -> None:
    """Upload a single file to RunPod S3 (per-file cp avoids pagination bug)."""
    dst = f"s3://{bucket}/{s3_key}"
    subprocess.run(["aws", "s3", "cp", local_path, dst, *_s3_base_args()],
                   capture_output=True)


def _hf_upload_file(local_path: str, repo_id: str, path_in_repo: str, commit_message: str) -> None:
    """Upload a single file to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    except Exception as e:
        print(f"  [bg] HF upload failed: {e}")


def background_upload(local_path: str, config: dict, experiment_name: str, label: str = "checkpoint") -> None:
    """Fire-and-forget upload of a checkpoint file to S3 and HF Hub.

    Runs in a daemon thread so it never blocks training.
    Note: if the file is overwritten before the upload finishes, the upload may fail.
    """
    xenc_cfg = config.get("train_x_encoder", {})
    s3_bucket = xenc_cfg.get("s3_bucket") or config.get("train_manifold_anchor", {}).get("s3_bucket")
    hf_repo = xenc_cfg.get("hf_repo") or config.get("train_manifold_anchor", {}).get("hf_repo")
    fname = os.path.basename(local_path)

    def _worker():
        if s3_bucket:
            s3_key = f"checkpoints/{experiment_name}/{fname}"
            _s3_upload_file(local_path, s3_bucket, s3_key)
        if hf_repo:
            _hf_upload_file(local_path, hf_repo, f"x_encoder/{experiment_name}/{fname}",
                            commit_message=f"[auto] {label}: {fname}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    _upload_threads.append(t)


def wait_for_uploads(timeout: float = 120.0) -> None:
    """Wait for all background uploads to finish (called at end of training)."""
    pending = [t for t in _upload_threads if t.is_alive()]
    if pending:
        print(f"Waiting for {len(pending)} background upload(s) to finish...")
        for t in pending:
            t.join(timeout=timeout)


def parse_cp_name(f):
    # example formats: "x_encoder_epoch_1.pt" or "x_encoder_epoch_1_step_10.pt"
    base = f.split('epoch_')[1].split('.pt')[0]
    if "_step_" in base:
        ep, st = base.split("_step_")
        return (int(ep), int(st))
    return (int(base), 0)

def parse_args():
    parser = argparse.ArgumentParser(description="LatentEuclid X-Encoder Full SFT Loop")
    parser.add_argument("--config", type=str, default="training/config.yaml",
                        help="Path to YAML training configuration")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Explicit experiment namespace override")
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
    
    experiment_name = args.experiment_name or config.get("train_x_encoder", {}).get("experiment_name") or config.get("experiment", {}).get("name", "default")
    
    # Pre-resolve Dynamic Namespaces for Transparent Telemetry Logging
    base_checkpoint_dir = config.get("train_x_encoder", {}).get("checkpoint_dir", "/workspace/checkpoints")
    checkpoint_dir = os.path.join(base_checkpoint_dir, experiment_name)
    config.setdefault("train_x_encoder", {})["checkpoint_dir"] = checkpoint_dir
    
    base_targets_dir = config.get("data", {}).get("targets_dir", "/workspace/target_tensors")
    targets_dir = os.path.join(base_targets_dir, f"target_tensors_{experiment_name}")
    config.setdefault("data", {})["targets_dir"] = targets_dir

    if is_master:
        print("\n" + "="*50)
        print("LatentEuclid Phase 4 (Continuous Alignment)")
        print("Executing with Configuration:")
        print(yaml.dump(config, default_flow_style=False))
        print("="*50 + "\n")
    
    if is_master:
        import time
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        wandb.init(
            project="LatentEuclid",
            name=f"{experiment_name}_XEncoder_{run_timestamp}",
            group=experiment_name,
            config=config
        )
    
    print(f"[{local_rank}] Instantiating LatentEuclid module constraints...")
    
    model = LatentEuclid(
        base_model_id=config["model"]["base_model_id"],
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    )
    
    # Activation Checkpointing trades 20-30% compute time for massive memory savings by dropping intermediate activations.
    model.vlm.gradient_checkpointing_enable()
    
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
    
    loss_target_mode = config["train_x_encoder"].get("loss_target", "guided")

    if is_distributed:
        criterion = criterion.to(local_rank)
        
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["train_x_encoder"]["learning_rate"]), 
        weight_decay=float(config["train_x_encoder"]["weight_decay"])
    )
    
    # ---------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    latest_cp_path = os.path.join(checkpoint_dir, "x_encoder_latest.pt")
    
    # Try loading latest checkpoint (consolidated single-file approach)
    if os.path.exists(latest_cp_path):
        print(f"[{local_rank}] Found x_encoder_latest.pt. Loading...")
        cp = torch.load(latest_cp_path, map_location="cpu")
        start_epoch = cp.get("epoch", 0)
        if "val_loss" in cp:
            best_val_loss = cp.get("val_loss", float('inf'))
        
        # Attempt to retrieve strictly tracked best_val_loss
        loss_tracker_path = os.path.join(checkpoint_dir, "best_val_loss.json")
        if os.path.exists(loss_tracker_path):
            with open(loss_tracker_path, 'r') as f:
                best_val_loss = float(json.load(f)["best_loss"])
        
        if is_distributed:
            model.module.load_state_dict(cp["model_state_dict"])
        else:
            model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        
        # Explicitly override the loaded learning rate with the config value
        target_lr = float(config["train_x_encoder"]["learning_rate"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr
        print(f"[{local_rank}] Resumed from epoch {start_epoch} | best_val_loss={best_val_loss:.4f} | LR overridden to {target_lr}")
    else:
        # Legacy fallback: find old epoch-based checkpoints
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("x_encoder_epoch_") and f.endswith(".pt")]
            if checkpoints:
                checkpoints.sort(key=parse_cp_name)
                legacy_cp_file = checkpoints[-1]
                legacy_cp_path = os.path.join(checkpoint_dir, legacy_cp_file)
                ep, st = parse_cp_name(legacy_cp_file)
                start_epoch = ep
                
                print(f"[{local_rank}] Found legacy checkpoint {legacy_cp_file}. Resuming from epoch {start_epoch}...")
                cp = torch.load(legacy_cp_path, map_location="cpu")
                if "val_loss" in cp:
                    best_val_loss = cp.get("val_loss", float('inf'))
                
                loss_tracker_path = os.path.join(checkpoint_dir, "best_val_loss.json")
                if os.path.exists(loss_tracker_path):
                    with open(loss_tracker_path, 'r') as f:
                        best_val_loss = float(json.load(f)["best_loss"])
                
                if is_distributed:
                    model.module.load_state_dict(cp["model_state_dict"])
                else:
                    model.load_state_dict(cp["model_state_dict"])
                optimizer.load_state_dict(cp["optimizer_state_dict"])
                
                target_lr = float(config["train_x_encoder"]["learning_rate"])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = target_lr
                print(f"[{local_rank}] Checkpoint Optimizer LR manually overridden to {target_lr}")
        else:
            if is_master:
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"[{local_rank}] Starting from scratch.")
    # ---------------------------------------------------------
    
    # Instantiate Data Loader
    full_dataset = GeoThoughtsDataset(
        jsonl_path=config["data"]["jsonl_path"],
        targets_dir=config["data"]["targets_dir"],
        tokenizer=model.module.tokenizer if is_distributed else model.tokenizer,
        k_steps=config["model"]["k_steps"],
        augment=config["train_x_encoder"].get("augment", False) # Read flag from config, default pure dataset
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
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=int(config["train_x_encoder"]["batch_size"]), 
        sampler=val_sampler,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True,
        shuffle=False
    )
    
    gradient_accumulation_steps = int(config["train_x_encoder"].get("gradient_accumulation_steps", 1))
    print(f"[{local_rank}] Successfully initialized epoch pipelines! Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    epochs = int(config["train_x_encoder"]["epochs"])
    max_steps_per_epoch = config["train_x_encoder"].get("max_steps_per_epoch", None)
    device = local_rank if is_distributed else local_rank
    
    # --- VALIDATION FUNCTION ---
    def run_validation(step_label, current_epoch, current_step):
        model.eval()
        total_val_loss = 0.0
        total_val_mse = 0.0
        val_samples_processed = 0
        max_val_samples = int(config["train_x_encoder"].get("max_val_samples", 0))
        
        if is_master:
            print(f"{step_label} | Running validation inference...")
            
        with torch.no_grad():
            for val_idx, (val_img, val_txt, val_targ) in enumerate(val_dataloader):
                # Stop early if we hit the requested validation subset size
                if max_val_samples > 0 and val_samples_processed >= max_val_samples:
                    break
                    
                current_batch_size = len(val_img)
                val_samples_processed += current_batch_size
                
                val_msgs = [
                    [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]
                    for img, txt in zip(val_img, val_txt)
                ]
                
                processor = model.module.processor if is_distributed else model.processor
                val_text_prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in val_msgs]
                val_inputs = processor(
                    text=val_text_prompts,
                    images=val_img,
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    val_pred = model(
                        input_ids=val_inputs.input_ids, 
                        attention_mask=val_inputs.attention_mask,
                        pixel_values=val_inputs.get("pixel_values"),
                        image_grid_thw=val_inputs.get("image_grid_thw")
                    )
                    val_targ = val_targ.to(device=device, dtype=val_pred.dtype)
                    v_loss, v_metrics = criterion(val_pred, val_targ)
                
                total_val_loss += v_loss.item()
                if "loss/cosine_angular" in v_metrics:
                    total_val_mse += v_metrics["loss/cosine_angular"]
                elif "loss/invariance_cos" in v_metrics:
                    total_val_mse += v_metrics["loss/invariance_cos"]
        
        # Note: val_idx corresponds to the number of batches actually processed (0-indexed)
        batches_processed = val_idx if (max_val_samples > 0 and val_samples_processed >= max_val_samples) else len(val_dataloader)
        avg_val_loss = total_val_loss / max(1, batches_processed)
        avg_val_mse = total_val_mse / max(1, batches_processed)
        
        if is_master:
            print(f"[{local_rank}] {step_label} Validation | Eval Samples: {val_samples_processed} | Avg Loss: {avg_val_loss:.4f} | Avg Cosine: {avg_val_mse:.4f}")
            wandb.log({"val/epoch_loss": avg_val_loss, "val/epoch_cos": avg_val_mse, "epoch": current_epoch, "step": current_step})
            
        model.train()
        return avg_val_loss
        
    # ------------------------------------

    for epoch in range(start_epoch, epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        avg_val_loss = float('inf')
        
        for batch_idx, (images, texts, targets) in enumerate(train_dataloader):
            micro_start_time = time.time()
            if max_steps_per_epoch is not None and batch_idx >= max_steps_per_epoch:
                print(f"[{local_rank}] Reached max_steps_per_epoch ({max_steps_per_epoch}). Ending epoch {epoch} early.")
                break
            if batch_idx % gradient_accumulation_steps == 0:
                step_start_time = time.time()
                
            # We extract them utilizing the associated model processor dynamically:
            processor = model.module.processor if is_distributed else model.processor
            
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": txt},
                        ],
                    }
                ]
                for img, txt in zip(images, texts)
            ]
            
            text_prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            inputs = processor(
                text=text_prompts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_latents = model(
                    input_ids=inputs.input_ids, 
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw")
                )
                
                targets = targets.to(device=device, dtype=predicted_latents.dtype)
                
                if loss_target_mode in ["direct", "pondering"]:
                    # Only calculate loss on the final step (thought_k)
                    predicted_latents_loss = predicted_latents[:, -1:, :]
                    targets_loss = targets[:, -1:, :]
                else:
                    predicted_latents_loss = predicted_latents
                    targets_loss = targets
                
                # Loss alignment mapping
                loss, metrics_dict = criterion(predicted_latents_loss, targets_loss)
                
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
                
                
                if is_master:
                    step_duration = time.time() - step_start_time
                    current_lr = optimizer.param_groups[0]['lr']
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    var_std_val = metrics_dict.get("loss/variance_std_physical", 0.0) if type(metrics_dict) is dict else 0.0
                    huber_val = metrics_dict.get("loss/huber_magnitude", 0.0) if type(metrics_dict) is dict else 0.0
                    train_mse_val = metrics_dict.get("loss/cosine_angular", metrics_dict.get("loss/invariance_cos", 0.0)) if type(metrics_dict) is dict else 0.0
                    print(f"Epoch {epoch} | Step {batch_idx + 1} | Time: {step_duration:.2f}s | Train Loss: {loss.item() * current_accumulation_steps:.4f} | Cos: {train_mse_val:.4f} | Huber: {huber_val:.4f} | Grad Norm: {grad_norm_val:.2f} | Var: {var_std_val:.3f}")
                    
                    # Push tracked metrics to WandB securely
                    metrics_dict["train/total_loss"] = loss.item() * current_accumulation_steps
                    metrics_dict["train/grad_norm"] = grad_norm_val
                    metrics_dict["train/learning_rate"] = current_lr
                    metrics_dict["epoch"] = epoch
                    
                    wandb.log(metrics_dict)
                    
                    # --- SAVE CHECKPOINT EVERY N STEPS ---
                    save_every_n_steps = int(config["train_x_encoder"].get("save_every_n_steps", 0))
                    global_step = (batch_idx + 1) // gradient_accumulation_steps
                    if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                        # Run Mid-Epoch Validation
                        mid_val_loss = run_validation(f"Mid-Epoch {epoch} (Step {global_step})", epoch, global_step)
                        
                        # Save single consolidated latest checkpoint (overwrites previous)
                        save_model = model.module if is_distributed else model
                        latest_save_path = os.path.join(checkpoint_dir, "x_encoder_latest.pt")
                        torch.save({
                            'epoch': epoch,
                            'step': global_step,
                            'model_state_dict': save_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                            'val_loss': mid_val_loss
                        }, latest_save_path)
                        print(f"[{local_rank}] Saved latest checkpoint: {latest_save_path}")
                        
                        # Upload latest to S3/HF in background
                        background_upload(latest_save_path, config, experiment_name, label="latest")
                        
                        # Update best if improved
                        if mid_val_loss < best_val_loss:
                            best_val_loss = mid_val_loss
                            best_cp_path = os.path.join(checkpoint_dir, "x_encoder_best.pt")
                            shutil.copy2(latest_save_path, best_cp_path)
                            
                            with open(os.path.join(checkpoint_dir, "best_val_loss.json"), 'w') as f:
                                json.dump({"best_loss": best_val_loss, "epoch": epoch, "step": global_step}, f)
                                
                            print(f"[{local_rank}] New best validation loss {best_val_loss:.4f}! Saved x_encoder_best.pt")
                            
                            # Upload best to S3/HF in background
                            background_upload(best_cp_path, config, experiment_name, label="best")
                            background_upload(os.path.join(checkpoint_dir, "best_val_loss.json"), config, experiment_name, label="best_meta")
                
            else:
                # Provide a live micro-batch progress indicator to the user
                micro_step = (batch_idx % gradient_accumulation_steps) + 1
                if is_master:
                    # Accurately extract the partial step metrics mapped from stable_alignment_loss
                    huber_val = metrics_dict.get("loss/huber_magnitude", 0.0) if type(metrics_dict) is dict else 0.0
                    train_mse_val = metrics_dict.get("loss/cosine_angular", metrics_dict.get("loss/invariance_cos", 0.0)) if type(metrics_dict) is dict else 0.0
                    
                    # Approximated gradient norm requires calculation prior to step if we want it continuously logged
                    temp_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train_x_encoder"]["max_grad_norm"])).item()
                    micro_duration = time.time() - micro_start_time
                    
                    print(f"Epoch {epoch} | Accumulating ({micro_step}/{gradient_accumulation_steps}) | Time: {micro_duration:.2f}s | Micro Loss: {loss.item() * current_accumulation_steps:.4f} | Cos: {train_mse_val:.4f} | Huber: {huber_val:.4f} | Temp Grad: {temp_grad_norm:.2f}", end='\r')
                    
                    # Push tracked partial metrics to WandB securely
                    current_lr = optimizer.param_groups[0]['lr']
                    # Use a separate wandb step metric so it doesn't overwrite the global accumulation steps
                    micro_metrics_dict = metrics_dict.copy()
                    micro_metrics_dict["train_micro/total_loss"] = loss.item() * current_accumulation_steps
                    micro_metrics_dict["train_micro/temp_grad_norm"] = temp_grad_norm
                    micro_metrics_dict["train_micro/learning_rate"] = current_lr
                    micro_metrics_dict["train_micro/step_time"] = micro_duration
                    micro_metrics_dict["epoch"] = epoch
                    
                    wandb.log(micro_metrics_dict)

                    wandb.log(micro_metrics_dict)

        # --- VALIDATION PER EPOCH ---
        if is_master:
            # Run Exhaustive End-of-Epoch Validation (skip if we just ran an exhaustive periodic step check)
            global_step_end = len(train_dataloader) // gradient_accumulation_steps
            save_every_n_steps = int(config["train_x_encoder"].get("save_every_n_steps", 0))
            if save_every_n_steps > 0 and (global_step_end % save_every_n_steps) <= 2:
                print(f"[{local_rank}] End-of-Epoch exactly abutts Step {global_step_end - 1}. Skipping redundant validation and blind checkpoint saving.")
            elif (epoch + 1) % 10 == 0:
                avg_val_loss = run_validation(f"End-of-Epoch {epoch}", epoch, global_step_end)
                
                # Save latest every 10 epochs
                save_model = model.module if is_distributed else model
                latest_save_path = os.path.join(checkpoint_dir, "x_encoder_latest.pt")
                torch.save({
                    'epoch': epoch + 1,  # next epoch to resume from
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'val_loss': avg_val_loss
                }, latest_save_path)
                print(f"[{local_rank}] Saved latest checkpoint (end of epoch {epoch})")
                background_upload(latest_save_path, config, experiment_name, label="latest")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_cp_path = os.path.join(checkpoint_dir, "x_encoder_best.pt")
                    shutil.copy2(latest_save_path, best_cp_path)
                    print(f"[{local_rank}] New best validation loss {best_val_loss:.4f}! Saved {best_cp_path}")
                    
                    loss_tracker_path = os.path.join(checkpoint_dir, "best_val_loss.json")
                    with open(loss_tracker_path, 'w') as f:
                        json.dump({"best_loss": best_val_loss, "epoch": epoch}, f)
                    
                    background_upload(best_cp_path, config, experiment_name, label="best")
                    background_upload(loss_tracker_path, config, experiment_name, label="best_meta")
            else:
                print(f"[{local_rank}] Epoch {epoch} — skipping checkpoint save (next save at epoch {((epoch // 10) + 1) * 10 - 1})")
    if is_master:
        save_model = model.module if is_distributed else model
        final_path = os.path.join(checkpoint_dir, "x_encoder_latest.pt")
        torch.save({
            'epoch': epochs,
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_path)
        print(f"Final model saved to {final_path}")
        background_upload(final_path, config, experiment_name, label="final")
        
        # Wait for all background uploads before exiting
        wait_for_uploads()
        print("All uploads complete.")
        wandb.finish()

if __name__ == "__main__":
    train()
