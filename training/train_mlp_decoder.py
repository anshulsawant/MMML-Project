import os, json, math, random, yaml, argparse, time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from tokenizers import Tokenizer
import sys
sys.path.append('data')
from evaluate_generated import safe_math_eval, normalize

from models.latent_euclid import LatentEuclid
from models.mlp_decoder import MLPDecoder
from training.train_decoder_projection import GeoThoughtsTextDataset, custom_collate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--experiment_name", type=str, default="mlp_math_decoder")
    return parser.parse_args()

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_in_memory(y_decoder, x_encoder, val_loader, device, tokenizer, limit=100):
    val_decoder = y_decoder.module if isinstance(y_decoder, DDP) else y_decoder
    val_encoder = x_encoder.module if isinstance(x_encoder, DDP) else x_encoder
    val_decoder.eval()
    val_encoder.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, texts, target_answers) in enumerate(val_loader):
            if total >= limit: break
            
            x_processor = val_encoder.processor
            batch_messages = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}] for img, txt in zip(images, texts)]
            rendered_texts = [x_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
            inputs = x_processor(text=rendered_texts, images=images, padding=True, return_tensors="pt").to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_latents = val_encoder(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get("pixel_values"), image_grid_thw=inputs.get("image_grid_thw")
                )
                logits = val_decoder(x=predicted_latents).logits
                preds = logits.argmax(dim=-1)
                
            for pred_ids, gt_ans in zip(preds, target_answers):
                clean_ids = []
                for i in pred_ids.tolist():
                    if i in [1, 3]: break # PAD=1, EOS=3
                    clean_ids.append(i)
                pred_raw = tokenizer.decode(clean_ids).replace(" ", "")
                gt_raw = gt_ans.replace("<|im_end|>", "").strip()
                
                if normalize(gt_raw) == normalize(pred_raw):
                    correct += 1
                else:
                    gt_val = safe_math_eval(gt_raw)
                    pred_val = safe_math_eval(pred_raw)
                    if gt_val is not None and pred_val is not None and math.isclose(gt_val, pred_val, rel_tol=1e-3, abs_tol=0.06):
                        correct += 1
                total += 1
                if total >= limit: break
                
    val_decoder.train()
    return (correct / total * 100) if total > 0 else 0.0

def train():
    args = parse_args()
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    local_rank = setup_ddp()
    is_master = (local_rank == 0 if isinstance(local_rank, int) else True)
    device = local_rank
    
    experiment_name = args.experiment_name
    checkpoint_dir = os.path.join("/workspace/checkpoints/mlp_decoder", experiment_name)
    if is_master: os.makedirs(checkpoint_dir, exist_ok=True)
    if is_master: wandb.init(project="LatentEuclid", name=experiment_name, config=config)

    x_encoder = LatentEuclid(base_model_id=config["model"]["base_model_id"], k_steps=config["model"]["k_steps"]).to(device)
    x_enc_weights = config.get("train_decoder", {}).get("x_encoder_weights_override", "")
    if os.path.exists(x_enc_weights):
        state = torch.load(x_enc_weights, map_location="cpu")
        x_encoder.load_state_dict(state.get("model_state_dict", state))
        if is_master: print(f"[{local_rank}] Loaded frozen X-Encoder from {x_enc_weights}")
    for p in x_encoder.parameters(): p.requires_grad = False
    x_encoder.eval()

    tokenizer = Tokenizer.from_file("data/math_vocab.json")
    y_decoder = MLPDecoder(input_dim=x_encoder.config.hidden_size, hidden_dim=2048, vocab_size=300, max_len=10, num_layers=4).to(device)
    
    if isinstance(local_rank, int):
        y_decoder = DDP(y_decoder, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(y_decoder.parameters(), lr=1e-4, weight_decay=0.01)

    with open(config["data"]["jsonl_path"], 'r') as f: full_data = [json.loads(line) for line in f]
    with open("data/ground_truths.json", 'r') as f: ground_truths = json.load(f)
        
    random.seed(42)
    random.shuffle(full_data)
    split_idx = int(0.9 * len(full_data))
    train_data, val_data = full_data[:split_idx], full_data[split_idx:]

    x_tokenizer = x_encoder.module.tokenizer if isinstance(local_rank, int) else x_encoder.tokenizer
    train_dataset = GeoThoughtsTextDataset(train_data, ground_truths, x_tokenizer, config["model"]["k_steps"])
    val_dataset = GeoThoughtsTextDataset(val_data, ground_truths, x_tokenizer, config["model"]["k_steps"])

    train_sampler = DistributedSampler(train_dataset) if isinstance(local_rank, int) else None
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=custom_collate, shuffle=(train_sampler is None), drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if isinstance(local_rank, int) else None
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler, collate_fn=custom_collate, shuffle=False)

    best_acc = 0.0
    for epoch in range(100):
        if train_sampler: train_sampler.set_epoch(epoch)
        y_decoder.train()
        
        for batch_idx, (images, texts, target_answers) in enumerate(train_loader):
            micro_start_time = time.time()
            optimizer.zero_grad()
            
            x_processor = x_encoder.module.processor if isinstance(local_rank, int) else x_encoder.processor
            batch_messages = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}] for img, txt in zip(images, texts)]
            rendered_texts = [x_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
            inputs = x_processor(text=rendered_texts, images=images, padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted_latents = x_encoder(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                        pixel_values=inputs.get("pixel_values"), image_grid_thw=inputs.get("image_grid_thw")
                    )

            encoded_targets = [tokenizer.encode(ans.replace("<|im_end|>", "").strip() + "[EOS]").ids for ans in target_answers]
            padded_targets = []
            for ids in encoded_targets:
                if len(ids) < 10: ids = ids + [1] * (10 - len(ids)) # PAD=1
                else: ids = ids[:10]
                padded_targets.append(ids)
            label_tensors = torch.tensor(padded_targets, dtype=torch.long, device=device)

            outputs = y_decoder(x=predicted_latents, labels=label_tensors)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(y_decoder.parameters(), 5.0)
            optimizer.step()
            
            if is_master:
                wandb.log({"train/loss": loss.item()})
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Time: {time.time()-micro_start_time:.2f}s", end='\r')

        if is_master:
            acc = evaluate_in_memory(y_decoder, x_encoder, val_loader, device, tokenizer, limit=200)
            wandb.log({"val/accuracy": acc})
            print(f"\n[Epoch {epoch}] Val Accuracy: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(y_decoder.module.state_dict() if isinstance(local_rank, int) else y_decoder.state_dict(), os.path.join(checkpoint_dir, "mlp_best.pt"))

if __name__ == "__main__":
    train()
