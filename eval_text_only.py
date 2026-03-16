import argparse
import yaml
import json
import torch
import math
import random
from tqdm import tqdm
import sys
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.append('data')
try:
    from evaluate_generated import clean_base_model_ans, safe_math_eval, normalize
except ImportError:
    print("Warning: Could not import robust evaluation metrics, defaulting to strict equality.")
    def clean_base_model_ans(x): return x.strip()
    def safe_math_eval(x): return None
    def normalize(x): return x.strip().lower()

def evaluate_text_only():
    parser = argparse.ArgumentParser(description="Zero-Shot Text-Only Evaluation")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--out", type=str, default="data/eval_text_only_baseline.json")
    args = parser.parse_args()

    device = args.device

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_id = config["model"]["target_model_id"]
    print(f"Loading Base Vision-Language Model for Text-Only Inference: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.eval()

    print("Loading Validation Dataset...")
    try:
        with open(config["data"]["jsonl_path"], 'r') as f:
            full_data = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading {config['data']['jsonl_path']}: {e}")
        return
        
    try:
        with open("data/ground_truths.json", 'r') as f:
            ground_truths = json.load(f)
    except Exception as e:
        print(f"Error loading ground_truths.json: {e}")
        return

    random.seed(42)
    random.shuffle(full_data)
    val_data = full_data[int(args.split * len(full_data)):]
    if args.limit:
        val_data = val_data[:args.limit]

    print(f"\n================ TEXT-ONLY ZERO-SHOT BASELINE ({len(val_data)} samples) ================\n")
    
    correct = 0
    total = 0
    results = []
    
    batches = [val_data[i:i + args.batch_size] for i in range(0, len(val_data), args.batch_size)]
    
    with torch.no_grad():
        pbar = tqdm(batches, desc="Evaluating Batches")
        for batch in pbar:
            batch_messages = []
            questions = []
            true_answers = []
            
            for item in batch:
                question = item["question"]
                true_answer_raw = ground_truths.get(item["image_path"])
                if true_answer_raw is None:
                    continue
                    
                messages = [{"role": "user", "content": [{"type": "text", "text": question + "\nAnswer: "}]}]
                batch_messages.append(messages)
                questions.append(question)
                true_answers.append((item["image_path"], true_answer_raw))
                
            if not batch_messages:
                continue
                
            rendered_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
            inputs = processor(text=rendered_texts, padding=True, return_tensors="pt").to(device)
            
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
            
            # Slice off the prompt logic correctly for Qwen models
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            generated_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            for gen_text, (img_path, true_answer_raw), q in zip(generated_texts, true_answers, questions):
                pred_raw = clean_base_model_ans(gen_text)
                gt_norm = normalize(true_answer_raw)
                pred_norm = normalize(pred_raw)
                
                is_correct = (gt_norm == pred_norm)
                if not is_correct:
                    gt_val = safe_math_eval(true_answer_raw)
                    pred_val = safe_math_eval(pred_raw)
                    if gt_val is not None and pred_val is not None:
                        is_correct = math.isclose(gt_val, pred_val, rel_tol=1e-3, abs_tol=0.06)
                        
                total += 1
                if is_correct:
                    correct += 1
                    
                results.append({
                    'image': img_path,
                    'is_correct': is_correct,
                    'gt_raw': str(true_answer_raw),
                    'pred_raw': pred_raw,
                    'gt_norm': gt_norm,
                    'pred_norm': pred_norm,
                    'model_generation': gen_text.strip()
                })
                    
            if total > 0:
                acc_so_far = correct / total * 100
                pbar.set_postfix({"Correct": f"{correct}/{total}", "Accuracy": f"{acc_so_far:.2f}%"})
                
            if results:
                with open(args.out, 'w') as f:
                    json.dump(results, f, indent=2)
                
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"\n+++ RESULTS +++")
    print(f"Evaluated {total} samples.")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    evaluate_text_only()
