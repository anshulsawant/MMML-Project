import argparse
import yaml
import json
import os
import shutil
import tempfile
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
# HuggingFace Hub background upload helpers
# ---------------------------------------------------------------------------

_upload_threads: list[threading.Thread] = []


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
    """Fire-and-forget upload of a checkpoint file to HuggingFace Hub.

    Runs in a daemon thread so it never blocks training.
    """
    xenc_cfg = config.get("train_x_encoder", {})
    hf_repo = xenc_cfg.get("hf_repo") or config.get("train_manifold_anchor", {}).get("hf_repo")
    fname = os.path.basename(local_path)

    def _worker():
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
    # example formats: "x_encoder_epoch_1.pt" or "x_encoder_epoch_1_step_10.pt" or "x_encoder_epoch_1_end.pt"
    base = f.split('epoch_')[1].split('.pt')[0].replace('_end', '')
    if "_step_" in base:
        ep, st = base.split("_step_")
        return (int(ep), int(st))
    return (int(base), 0)

def atomic_torch_save(state_dict, file_path):
    """
    Atomically saves a checkpoint by writing to a temporary file first, then
    copying to the final destination. This prevents partial/corrupt checkpoint
    files if the process is interrupted mid-write.
    Uses /dev/shm (shared memory) when available for faster intermediate I/O,
    falling back to the system temp directory.
    """
    # Write to fast temp location first
    temp_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "latent_euclid_checkpoint_tmp.pt")

    torch.save(state_dict, temp_path)
    shutil.copyfile(temp_path, file_path)

    if os.path.exists(temp_path):
        os.remove(temp_path)

def run_geometry_sanity_check(
    model,
    criterion,
    full_dataset,
    xenc_cfg,
    loss_target_mode,
    device,
    is_distributed,
    is_master,
    local_rank,
):
    """
    Single-batch overfit test: verifies γ is balanced so the contrastive loss
    does not dominate and collapse spatial geometry.

    Protocol
    --------
    1. Draw a fixed batch of 32 samples from the dataset (text-only; no images).
    2. For each sample append <thought_1>…<thought_N> tokens (N = # target steps).
    3. Run the composite loss (spatial + γ·InfoNCE) for SANITY_STEPS=50 gradient steps.
    4. Both losses should smoothly approach zero (single-batch overfit).
    5. Model and criterion weights are saved before and fully restored after — training
       state is completely unaffected.

    Logged to WandB under the 'sanity/' prefix so you can inspect the curves.
    """
    SANITY_N = 32
    SANITY_STEPS = 50

    if is_master:
        print("\n" + "=" * 60)
        print("  Geometry Sanity Check  (γ calibration · text-only · 50 steps)")
        print("=" * 60)

    # ── 1. Fixed 32-sample batch ──────────────────────────────────────────
    n_samples = min(SANITY_N, len(full_dataset))
    sanity_subset = torch.utils.data.Subset(full_dataset, list(range(n_samples)))
    sanity_loader = DataLoader(
        sanity_subset, batch_size=n_samples, collate_fn=custom_collate, shuffle=False
    )
    batch = next(iter(sanity_loader))
    cod_texts   = batch["cod_texts"]
    cot_texts   = batch["cot_texts"]
    targets     = batch["targets"].to(device)      # [B, N, dim]
    target_mask = batch["target_mask"].to(device)  # [B, N]

    # ── 2. Text-only inputs: append <thought_k> tokens per sample ─────────
    #    The main forward() locates thought tokens by id — no image needed.
    inner_model = model.module if is_distributed else model
    tokenizer   = inner_model.tokenizer

    text_inputs = []
    for cod_text, mask in zip(cod_texts, target_mask):
        n_thoughts  = int(mask.sum().item())
        thought_str = "".join(f"<thought_{k + 1}>" for k in range(n_thoughts))
        text_inputs.append(f"{cod_text} {thought_str}")

    encodings = tokenizer(
        text_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # ── 3. Save model + criterion state (restored at the end) ─────────────
    saved_model_state     = {k: v.clone() for k, v in model.state_dict().items()}
    saved_criterion_state = {k: v.clone() for k, v in criterion.state_dict().items()}

    sanity_optim = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=float(xenc_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(xenc_cfg.get("weight_decay", 0.01)),
    )

    model.train()
    criterion.train()

    metrics = {}
    loss    = torch.tensor(0.0)

    # ── 4. 50-step single-batch overfit loop ──────────────────────────────
    for step in range(SANITY_STEPS):
        sanity_optim.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predicted_latents = model(
                input_ids        =encodings["input_ids"],
                attention_mask   =encodings["attention_mask"],
                pixel_values     =None,
                image_grid_thw   =None,
                mm_token_type_ids=None,
            )

            Z_cod, Z_cot = inner_model.forward_contrastive_texts(cod_texts, cot_texts)

            tgt = targets.to(dtype=predicted_latents.dtype)
            if loss_target_mode in ["direct", "pondering"]:
                pred_l = torch.stack(
                    [p[m.sum() - 1] for p, m in zip(predicted_latents, target_mask)]
                ).unsqueeze(1)
                targ_l = torch.stack(
                    [t[m.sum() - 1] for t, m in zip(tgt, target_mask)]
                ).unsqueeze(1)
            else:
                pred_l = predicted_latents[target_mask].unsqueeze(1)
                targ_l = tgt[target_mask].unsqueeze(1)

            loss, metrics = criterion(pred_l, targ_l, Z_cod=Z_cod, Z_cot=Z_cot)

        loss.backward()
        sanity_optim.step()

        if is_master:
            huber_val       = metrics.get("loss/huber_magnitude", 0.0)
            contrastive_val = metrics.get("loss/contrastive",     0.0)
            total_val       = loss.item()
            print(
                f"  [{step + 1:2d}/{SANITY_STEPS}] "
                f"total={total_val:.4f}  huber={huber_val:.4f}  contrastive={contrastive_val:.4f}"
            )
            if wandb.run is not None:
                wandb.log({
                    "sanity/total_loss":       total_val,
                    "sanity/huber_loss":       huber_val,
                    "sanity/contrastive_loss": contrastive_val,
                    "sanity/step":             step,
                })

    # ── 5. Verdict ─────────────────────────────────────────────────────────
    if is_master:
        final_huber       = metrics.get("loss/huber_magnitude", 0.0)
        final_contrastive = metrics.get("loss/contrastive",     0.0)
        print()
        if final_huber > 0.5:
            print(
                f"  [WARN] Huber={final_huber:.4f} did not converge below 0.5. "
                "γ may be too large — spatial geometry is being overwhelmed by contrastive loss."
            )
        elif final_contrastive > 0.3:
            print(
                f"  [WARN] Contrastive={final_contrastive:.4f} did not converge below 0.3. "
                "Check InfoNCE temperature or whether CoD/CoT texts are too similar."
            )
        else:
            print(
                f"  [OK] Both losses converged  "
                f"(huber={final_huber:.4f}, contrastive={final_contrastive:.4f}). "
                "γ looks well-tuned — proceed with training."
            )
        print("=" * 60 + "\n")

    # ── 6. Restore pristine weights ────────────────────────────────────────
    model.load_state_dict(saved_model_state)
    criterion.load_state_dict(saved_criterion_state)
    model.train()
    criterion.train()


def parse_args():
    parser = argparse.ArgumentParser(description="LatentEuclid X-Encoder Full SFT Loop")
    parser.add_argument("--config", type=str, default="configs/v12_cod.yaml",
                        help="Path to YAML training configuration")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Explicit experiment namespace override")
    parser.add_argument("--config_block", type=str, default=None,
                        help="Optional x-encoder config block override")
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
    def __init__(
        self,
        jsonl_path: str,
        targets_dir: str,
        augment=False,
        targets_hf_repo: str | None = None,
        targets_hf_prefixes: list[str] | None = None,
    ):
        self.data = []
        self.targets_dir = targets_dir
        self.augmentor = GeometrySafeAugmentation() if augment else None
        self.targets_hf_repo = targets_hf_repo
        # Ordered list of folder paths inside the HF repo to search for tensors.
        self.targets_hf_prefixes: list[str] = [
            p.strip("/") for p in (targets_hf_prefixes or [])
        ]

        os.makedirs(self.targets_dir, exist_ok=True)

        local_count = 0
        remote_candidate_count = 0
        skipped_count = 0
        
        with open(jsonl_path, 'r') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                
                # Local-first: keep local tensor paths when already present.
                target_path = os.path.join(targets_dir, f"problem_{idx}_targets.pt")
                if os.path.exists(target_path):
                    item["target_path"] = target_path
                    self.data.append(item)
                    local_count += 1
                elif self.targets_hf_repo:
                    # Keep sample; we'll lazily fetch tensor from HF in __getitem__.
                    item["target_idx"] = idx
                    self.data.append(item)
                    remote_candidate_count += 1
                else:
                    skipped_count += 1
                    
        print(
            f"Loaded {len(self.data)} aligned datasets "
            f"(local={local_count}, hf_fallback={remote_candidate_count}, skipped={skipped_count})."
        )

    def _resolve_target_path(self, item: dict) -> str:
        """Resolve target tensor path with local-first, HF fallback behavior."""
        local_path = item.get("target_path")
        if local_path and os.path.exists(local_path):
            return local_path

        target_idx = item.get("target_idx")
        if target_idx is None:
            raise FileNotFoundError("Missing target path and target_idx for sample")

        expected_local = os.path.join(self.targets_dir, f"problem_{target_idx}_targets.pt")
        if os.path.exists(expected_local):
            item["target_path"] = expected_local
            return expected_local

        if not self.targets_hf_repo:
            raise FileNotFoundError(f"Target tensor not found locally: {expected_local}")

        filename = f"problem_{target_idx}_targets.pt"
        prefixes = self.targets_hf_prefixes or [""]  # try repo root if no prefixes given

        from huggingface_hub import hf_hub_download

        last_exc: Exception = FileNotFoundError("no prefixes to try")
        for prefix in prefixes:
            repo_filename = f"{prefix}/{filename}" if prefix else filename
            try:
                downloaded_path = hf_hub_download(
                    repo_id=self.targets_hf_repo,
                    filename=repo_filename,
                    repo_type="model",
                    local_dir=self.targets_dir,
                )
                item["target_path"] = downloaded_path
                return downloaded_path
            except Exception as e:
                last_exc = e

        raise FileNotFoundError(
            f"Target tensor not found locally ({expected_local}) and HF download failed across "
            f"all prefixes {prefixes} in {self.targets_hf_repo}: {last_exc}"
        ) from last_exc

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
            image = Image.new('RGB', (224, 224), color=(73, 109, 137))

        # 2. Text sequences for dual-alignment objective
        cod_steps = item.get("CoD_steps", [])
        cod_text = " ".join(cod_steps) if isinstance(cod_steps, list) else str(cod_steps)
        cot_text = item.get("CoT_text", "")

        # 3. Target manifolds [N, target_dim]
        target_path = self._resolve_target_path(item)
        target_tensor = torch.load(target_path, map_location="cpu", weights_only=True)

        return {
            "image": image,
            "cod_text": cod_text,
            "cot_text": cot_text,
            "target": target_tensor,
        }
        
from torch.nn.utils.rnn import pad_sequence

def custom_collate(batch):
    images = [item["image"] for item in batch]
    cod_texts = [item["cod_text"] for item in batch]
    cot_texts = [item["cot_text"] for item in batch]
    targets = [item["target"] for item in batch]

    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0.0)

    target_mask = torch.zeros(len(targets), targets_padded.size(1), dtype=torch.bool)
    for i, t in enumerate(targets):
        target_mask[i, :len(t)] = True

    return {
        "images": images,
        "cod_texts": cod_texts,
        "cot_texts": cot_texts,
        "targets": targets_padded,
        "target_mask": target_mask,
    }

def train():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    local_rank = setup_ddp()
    is_distributed = isinstance(local_rank, int)
    is_master = (local_rank == 0 if is_distributed else True)

    active_block = "train_x_encoder"
    if args.config_block:
        active_block = args.config_block
    elif args.experiment_name:
        if args.experiment_name in config and isinstance(config[args.experiment_name], dict):
            active_block = args.experiment_name
        else:
            for block_name, block_cfg in config.items():
                if not isinstance(block_cfg, dict):
                    continue
                if block_name == "train_x_encoder" or block_name.startswith("train_x_encoder_"):
                    if block_cfg.get("experiment_name") == args.experiment_name:
                        active_block = block_name
                        break

    xenc_cfg = dict(config.get("train_x_encoder", {}))
    if active_block != "train_x_encoder":
        xenc_cfg.update(config.get(active_block, {}))
    config["train_x_encoder"] = xenc_cfg
    
    experiment_name = args.experiment_name or xenc_cfg.get("experiment_name") or config.get("experiment", {}).get("name", "default")
    
    # Pre-resolve Dynamic Namespaces for Transparent Telemetry Logging
    base_checkpoint_dir = xenc_cfg.get("checkpoint_dir", "./checkpoints")
    checkpoint_dir = os.path.join(base_checkpoint_dir, experiment_name)
    config.setdefault("train_x_encoder", {})["checkpoint_dir"] = checkpoint_dir
    
    base_targets_dir = config.get("data", {}).get("targets_dir", "./target_tensors")
    targets_dir = os.path.join(base_targets_dir, f"target_tensors_{experiment_name}")
    config.setdefault("data", {})["targets_dir"] = targets_dir

    if is_master:
        print("\n" + "="*50)
        print("LatentEuclid Phase 4 (Continuous Alignment)")
        print("Executing with Configuration:")
        print(yaml.dump(config, default_flow_style=False))
        print("="*50 + "\n")
    
    wandb_enabled = bool(config.get("wandb", {}).get("enabled", True))
    wandb_active = False
    if is_master and wandb_enabled:
        import time

        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        git_branch = os.getenv("GIT_BRANCH", "unknown")
        wandb_cfg = config.get("wandb", {})
        wandb_project = wandb_cfg.get("project") or os.getenv("WANDB_PROJECT", "LatentEuclid")
        wandb_group = wandb_cfg.get("group") or os.getenv("WANDB_RUN_GROUP", f"{experiment_name}_{git_branch}")
        run_name_prefix = wandb_cfg.get("name_prefix", f"{experiment_name}_{git_branch}_XEncoder")

        wandb.init(
            project=wandb_project,
            name=f"{run_name_prefix}_{run_timestamp}",
            group=wandb_group,
            config=config,
        )
        wandb_active = True
    
    print(f"[{local_rank}] Instantiating LatentEuclid module constraints...")
    
    model = LatentEuclid(
        base_model_id=config["model"]["base_model_id"],
        target_model_id=config["model"]["target_model_id"]
    )
    
    # Activation Checkpointing trades 20-30% compute time for massive memory savings by dropping intermediate activations.
    model.vlm.gradient_checkpointing_enable()
    
    if is_distributed:
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        model = model.to(local_rank) # cpu/cuda

    model_ref = model.module if is_distributed else model
    model_hidden_dim = int(getattr(getattr(model_ref, "config", object()), "hidden_size", 3584))
    contrastive_hidden_dim = int(xenc_cfg.get("hidden_dim", model_hidden_dim))
    contrastive_queue_size = int(xenc_cfg.get("queue_size", 128))
        
    criterion = AlignmentLossFactory(
        loss_type=xenc_cfg["loss_type"],
        vicreg_sim_coeff=float(xenc_cfg.get("vicreg_sim_coeff", 25.0)),
        vicreg_var_coeff=float(xenc_cfg.get("vicreg_var_coeff", 25.0)),
        vicreg_cov_coeff=float(xenc_cfg.get("vicreg_cov_coeff", 1.0)),
        gamma=float(xenc_cfg.get("gamma", 0.0)),
        queue_size=contrastive_queue_size,
        hidden_dim=contrastive_hidden_dim,
    )
    criterion = criterion.to(local_rank)
    
    loss_target_mode = xenc_cfg.get("loss_target", "guided")
        
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(xenc_cfg["learning_rate"]), 
        weight_decay=float(xenc_cfg["weight_decay"])
    )
    
    # ---------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    use_robust_checkpoint_save = bool(xenc_cfg.get("use_robust_checkpoint_save", True))
    keep_step_checkpoints = int(xenc_cfg.get("keep_step_checkpoints", 4))
    keep_epoch_checkpoints = int(xenc_cfg.get("keep_epoch_checkpoints", 4))
    save_epoch_checkpoints_after = int(xenc_cfg.get("save_epoch_checkpoints_after", 7))

    def persist_checkpoint(payload: dict, save_path: str) -> None:
        if use_robust_checkpoint_save:
            atomic_torch_save(payload, save_path)
        else:
            torch.save(payload, save_path)

    def prune_checkpoint_family(prefix: str, suffix: str, keep: int) -> None:
        if keep <= 0:
            return
        existing = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(suffix)]
        if len(existing) <= keep:
            return
        existing.sort(key=parse_cp_name)
        for old_f in existing[:-keep]:
            try:
                os.remove(os.path.join(checkpoint_dir, old_f))
            except OSError:
                pass

    def update_best_checkpoint(source_path: str, val_loss: float, epoch: int, step: int | None = None) -> None:
        nonlocal best_val_loss
        if val_loss >= best_val_loss:
            return

        best_val_loss = val_loss
        best_cp_path = os.path.join(checkpoint_dir, "x_encoder_best.pt")
        shutil.copy2(source_path, best_cp_path)

        tracker_payload = {"best_loss": best_val_loss, "epoch": epoch}
        if step is not None:
            tracker_payload["step"] = step

        loss_tracker_path = os.path.join(checkpoint_dir, "best_val_loss.json")
        with open(loss_tracker_path, 'w') as f:
            json.dump(tracker_payload, f)

        step_suffix = f" step {step}" if step is not None else ""
        print(f"[{local_rank}] New best validation loss {best_val_loss:.4f} at epoch {epoch}{step_suffix}. Saved {best_cp_path}")
        background_upload(best_cp_path, config, experiment_name, label="best")
        background_upload(loss_tracker_path, config, experiment_name, label="best_meta")
    
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
        target_lr = float(xenc_cfg["learning_rate"])
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
                
                target_lr = float(xenc_cfg["learning_rate"])
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
        augment=xenc_cfg.get("augment", False),
        targets_hf_repo=(
            config.get("data", {}).get("targets_hf_repo")
            or xenc_cfg.get("targets_hf_repo")
        ),
        targets_hf_prefixes=(
            config.get("data", {}).get("targets_hf_prefixes")
            or xenc_cfg.get("targets_hf_prefixes")
        ),
    )
    
    # V4 Aligned Deterministic Extracted Splits tracking precise topological boundaries naturally
    def _fallback_random_split(reason: str):
        if is_master:
            print(f"X-Encoder falling back to legacy 90-10 random splits: {reason}")
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        return torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    try:
        with open("data/v4_split_keys.json", "r") as f:
            v4_splits = json.load(f)
        v4_val_keys = set(v4_splits["val_keys"])
        v4_train_keys = set(v4_splits["train_keys"])
        
        train_indices = []
        val_indices = []
        for i, item in enumerate(full_dataset.data):
            raw_path = item["image_path"]
            norm_path = raw_path.lstrip("./")
            base = os.path.basename(norm_path)
            path_candidates = {raw_path, norm_path, base, f"./{norm_path}"}

            if any(k in v4_val_keys for k in path_candidates):
                val_indices.append(i)
            elif any(k in v4_train_keys for k in path_candidates):
                train_indices.append(i)

        if len(train_indices) == 0 or len(val_indices) == 0:
            train_dataset, val_dataset = _fallback_random_split(
                f"V4 key mapping produced empty split (train={len(train_indices)}, val={len(val_indices)})"
            )
        else:
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
            if is_master:
                print(
                    f"X-Encoder mapped explicit V4 aligned boundaries! "
                    f"{len(train_indices)} train | {len(val_indices)} val keys isolated."
                )
    except Exception as e:
        train_dataset, val_dataset = _fallback_random_split(str(e))
    
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
        max_val_samples = int(xenc_cfg.get("max_val_samples", 0))
        
        if is_master:
            print(f"{step_label} | Running validation inference...")
            
        with torch.no_grad():
            for val_idx, val_batch in enumerate(val_dataloader):
                val_img = val_batch["images"]
                val_txt = val_batch["cod_texts"]
                val_targ = val_batch["targets"]
                val_target_mask = val_batch["target_mask"]
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
                        image_grid_thw=val_inputs.get("image_grid_thw"),
                        mm_token_type_ids=val_inputs.get("mm_token_type_ids")
                    )
                    val_targ = val_targ.to(device=device, dtype=val_pred.dtype)
                    
                    if loss_target_mode in ["direct", "pondering"]:
                        # Extract the exact final step natively ignoring arbitrary lengths natively
                        val_pred_loss = torch.stack([pred[m.sum()-1] for pred, m in zip(val_pred, val_target_mask)])
                        val_targ_loss = torch.stack([targ[m.sum()-1] for targ, m in zip(val_targ, val_target_mask)])
                        # Expand functionally for criterion input dimensions seamlessly mapping to flattened states
                        val_pred_loss = val_pred_loss.unsqueeze(1)
                        val_targ_loss = val_targ_loss.unsqueeze(1)
                    else:
                        val_pred_loss = val_pred[val_target_mask]
                        val_targ_loss = val_targ[val_target_mask]
                        val_pred_loss = val_pred_loss.unsqueeze(1)
                        val_targ_loss = val_targ_loss.unsqueeze(1)
                        
                    v_loss, v_metrics = criterion(val_pred_loss, val_targ_loss)
                
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
            if wandb_active:
                wandb.log({"val/epoch_loss": avg_val_loss, "val/epoch_cos": avg_val_mse, "epoch": current_epoch, "step": current_step})
            
        model.train()
        return avg_val_loss
        
    # ------------------------------------

    # Optional γ-calibration sanity check (set sanity_check: true in config to enable)
    if is_master and xenc_cfg.get("sanity_check", False):
        run_geometry_sanity_check(
            model=model,
            criterion=criterion,
            full_dataset=full_dataset,
            xenc_cfg=xenc_cfg,
            loss_target_mode=loss_target_mode,
            device=device,
            is_distributed=is_distributed,
            is_master=is_master,
            local_rank=local_rank,
        )

    for epoch in range(start_epoch, epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        avg_val_loss = float('inf')
        
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch["images"]
            cod_texts = batch["cod_texts"]
            cot_texts = batch["cot_texts"]
            targets = batch["targets"]
            target_masks = batch["target_mask"]
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
                for img, txt in zip(images, cod_texts)
            ]
            
            text_prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            inputs = processor(
                text=text_prompts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            if batch_idx == 0 and is_master:
                pixel_val_shape = inputs.get("pixel_values").shape if inputs.get("pixel_values") is not None else "None !!"
                print(f"\n[Hardware Matrix Sanity Check] Successfully routed batch 0 images! pixel_values shape: {pixel_val_shape}")
                
            inner_model = model.module if is_distributed else model

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_latents = model(
                    input_ids=inputs.input_ids, 
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    mm_token_type_ids=inputs.get("mm_token_type_ids")
                )

                # Dual-stream contrastive encoding: CoD and CoT text → fixed-size embeddings
                Z_cod, Z_cot = inner_model.forward_contrastive_texts(cod_texts, cot_texts)

                targets = targets.to(device=device, dtype=predicted_latents.dtype)
                
                if loss_target_mode in ["direct", "pondering"]:
                    # Dynamically extract HALT target mathematically
                    predicted_latents_loss = torch.stack([pred[m.sum()-1] for pred, m in zip(predicted_latents, target_masks)])
                    targets_loss = torch.stack([targ[m.sum()-1] for targ, m in zip(targets, target_masks)])
                    predicted_latents_loss = predicted_latents_loss.unsqueeze(1)
                    targets_loss = targets_loss.unsqueeze(1)
                else:
                    # Globally evaluate purely over dimensions valid to unroll natively!
                    predicted_latents_loss = predicted_latents[target_masks]
                    targets_loss = targets[target_masks]
                    # Map onto sequence abstraction array logic uniformly
                    predicted_latents_loss = predicted_latents_loss.unsqueeze(1)
                    targets_loss = targets_loss.unsqueeze(1)
                
                # Composite multi-objective loss: spatial alignment + semantic contrastive regularization
                loss, metrics_dict = criterion(predicted_latents_loss, targets_loss, Z_cod=Z_cod, Z_cot=Z_cot)
                
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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(xenc_cfg["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad()
                
                
                if is_master:
                    step_duration = time.time() - step_start_time
                    current_lr = optimizer.param_groups[0]['lr']
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    var_std_val = metrics_dict.get("loss/variance_std_physical", 0.0) if type(metrics_dict) is dict else 0.0
                    huber_val = metrics_dict.get("loss/huber_magnitude", 0.0) if type(metrics_dict) is dict else 0.0
                    train_mse_val = metrics_dict.get("loss/cosine_angular", metrics_dict.get("loss/invariance_cos", 0.0)) if type(metrics_dict) is dict else 0.0
                    contrastive_val = metrics_dict.get("loss/contrastive", 0.0) if type(metrics_dict) is dict else 0.0
                    print(f"Epoch {epoch} | Step {batch_idx + 1} | Time: {step_duration:.2f}s | Train Loss: {loss.item() * current_accumulation_steps:.4f} | Cos: {train_mse_val:.4f} | Huber: {huber_val:.4f} | Contrastive: {contrastive_val:.4f} | Grad Norm: {grad_norm_val:.2f} | Var: {var_std_val:.3f}")
                    
                    # Push tracked metrics to WandB securely
                    metrics_dict["train/total_loss"] = loss.item() * current_accumulation_steps
                    metrics_dict["train/grad_norm"] = grad_norm_val
                    metrics_dict["train/learning_rate"] = current_lr
                    metrics_dict["train/contrastive_loss"] = contrastive_val
                    metrics_dict["epoch"] = epoch
                    
                    if wandb_active:
                        wandb.log(metrics_dict)
                    
                    # --- SAVE CHECKPOINT EVERY N STEPS ---
                    save_every_n_steps = int(xenc_cfg.get("save_every_n_steps", 0))
                    global_step = (batch_idx + 1) // gradient_accumulation_steps
                    if save_every_n_steps > 0 and global_step % save_every_n_steps == 0:
                        # Run Mid-Epoch Validation
                        mid_val_loss = run_validation(f"Mid-Epoch {epoch} (Step {global_step})", epoch, global_step)
                        save_model = model.module if is_distributed else model
                        checkpoint_payload = {
                            'epoch': epoch,
                            'step': global_step,
                            'model_state_dict': save_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                            'val_loss': mid_val_loss
                        }

                        latest_save_path = os.path.join(checkpoint_dir, "x_encoder_latest.pt")
                        persist_checkpoint(checkpoint_payload, latest_save_path)
                        print(f"[{local_rank}] Saved latest checkpoint: {latest_save_path}")

                        # Upload latest to S3/HF in background
                        background_upload(latest_save_path, config, experiment_name, label="latest")

                        if keep_step_checkpoints > 0:
                            step_cp_path = os.path.join(checkpoint_dir, f"x_encoder_epoch_{epoch}_step_{global_step}.pt")
                            persist_checkpoint(checkpoint_payload, step_cp_path)
                            prune_checkpoint_family(f"x_encoder_epoch_{epoch}_step_", ".pt", keep_step_checkpoints)

                        update_best_checkpoint(latest_save_path, mid_val_loss, epoch, step=global_step)
                
            else:
                # Provide a live micro-batch progress indicator to the user
                micro_step = (batch_idx % gradient_accumulation_steps) + 1
                if is_master:
                    # Accurately extract the partial step metrics mapped from stable_alignment_loss
                    huber_val = metrics_dict.get("loss/huber_magnitude", 0.0) if type(metrics_dict) is dict else 0.0
                    train_mse_val = metrics_dict.get("loss/cosine_angular", metrics_dict.get("loss/invariance_cos", 0.0)) if type(metrics_dict) is dict else 0.0
                    
                    # Approximated gradient norm requires calculation prior to step if we want it continuously logged
                    temp_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(xenc_cfg["max_grad_norm"])).item()
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
                    
                    if wandb_active:
                        wandb.log(micro_metrics_dict)

        # --- VALIDATION PER EPOCH ---
        if is_master:
            # Run Exhaustive End-of-Epoch Validation (skip if we just ran an exhaustive periodic step check)
            global_step_end = len(train_dataloader) // gradient_accumulation_steps
            save_every_n_steps = int(xenc_cfg.get("save_every_n_steps", 0))
            if save_every_n_steps > 0 and (global_step_end % save_every_n_steps) <= 2:
                print(f"[{local_rank}] End-of-Epoch exactly abutts Step {global_step_end - 1}. Skipping redundant validation and blind checkpoint saving.")
            elif (epoch + 1) % 10 == 0:
                avg_val_loss = run_validation(f"End-of-Epoch {epoch}", epoch, global_step_end)

                save_model = model.module if is_distributed else model
                checkpoint_payload = {
                    'epoch': epoch + 1,
                    'step': global_step_end,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'val_loss': avg_val_loss
                }

                latest_save_path = os.path.join(checkpoint_dir, "x_encoder_latest.pt")
                persist_checkpoint(checkpoint_payload, latest_save_path)
                print(f"[{local_rank}] Saved latest checkpoint (end of epoch {epoch})")
                background_upload(latest_save_path, config, experiment_name, label="latest")

                if keep_epoch_checkpoints > 0 and epoch >= save_epoch_checkpoints_after:
                    epoch_cp_path = os.path.join(checkpoint_dir, f"x_encoder_epoch_{epoch}_end.pt")
                    historical_payload = dict(checkpoint_payload)
                    historical_payload['epoch'] = epoch
                    persist_checkpoint(historical_payload, epoch_cp_path)
                    print(f"[{local_rank}] Saved end-of-epoch checkpoint: {epoch_cp_path}")
                    prune_checkpoint_family("x_encoder_epoch_", "_end.pt", keep_epoch_checkpoints)

                update_best_checkpoint(latest_save_path, avg_val_loss, epoch)
            else:
                print(f"[{local_rank}] Epoch {epoch} — skipping checkpoint save (next save at epoch {((epoch // 10) + 1) * 10 - 1})")
    if is_master:
        save_model = model.module if is_distributed else model
        final_path = os.path.join(checkpoint_dir, "x_encoder_latest.pt")
        persist_checkpoint({
            'epoch': epochs,
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_path)
        print(f"Final model saved to {final_path}")
        background_upload(final_path, config, experiment_name, label="final")

        final_state_path = os.path.join(checkpoint_dir, "latent_euclid_x_encoder_final.pt")
        persist_checkpoint(save_model.state_dict(), final_state_path)
        print(f"Model state successfully saved to {final_state_path}")

        # Wait for all background uploads before exiting
        wait_for_uploads()
        print("All uploads complete.")
        if wandb_active:
            wandb.finish()

if __name__ == "__main__":
    train()
