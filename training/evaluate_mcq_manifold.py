import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import argparse
import os
from tqdm import tqdm

from models.latent_euclid import LatentEuclid
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/config.yaml",
                        help="Path to YAML training configuration")
    parser.add_argument("--x_encoder_weights", type=str, default="checkpoints/x_encoder_best.pt",
                        help="Path to the frozen VICReg-aligned X-Encoder weights")
    return parser.parse_args()

class GeoThoughtsMCQDataset(Dataset):
    def __init__(self, data_list, ground_truths, k_steps=4):
        self.data = data_list
        self.ground_truths = ground_truths
        self.k_steps = k_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_path = item["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), color = (73, 109, 137))
            
        thought_string = "".join([f"<thought_{i+1}>" for i in range(self.k_steps)])
        full_text = item["question"] + " " + thought_string
        
        # Parse out the A, B, C, D options from the question string
        # Format: Options:\nA) 10\nB) 15\nC) 20\nD) 25
        options_text = item["question"].split("Options:\n")[1]
        lines = options_text.split('\n')
        
        option_map = {}
        for line in lines[:4]:
            if len(line) > 3 and line[1] == ')':
                letter = line[0]
                text_val = line[3:].strip()
                option_map[letter] = text_val
                
        target_letter = str(self.ground_truths.get(img_path, "A")).replace("<|im_end|>", "").strip()

        return {
            "image": image,
            "text": full_text,
            "target_letter": target_letter,
            "option_map": option_map
        }

def custom_collate(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    targets = [item["target_letter"] for item in batch]
    maps = [item["option_map"] for item in batch]
    return images, texts, targets, maps

def get_y_encoder_embedding(y_encoder, y_tokenizer, text_str, device):
    """
    Extract the mean-pooled vector embeddings for a specific option string
    from the frozen Y-Encoder (Qwen 0.6B).
    """
    inputs = y_tokenizer(text_str, return_tensors="pt").to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = y_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # Use the mean-pooled last hidden state of the text decoder as its representation
            hidden = outputs.hidden_states[-1][0] # [seq_len, dim]
            
            # Mask out padding theoretically if batching, but we do bs=1 here so mean is safe
            mean_hidden = hidden.mean(dim=0)
    return mean_hidden

def run_evaluation():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Zero-Shot Manifold Evaluation on Device: {device}")
    
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    base_model_id = config.get("model", {}).get("base_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    target_model_id = config.get("model", {}).get("target_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    decoder_base_model_id = config.get("model", {}).get("decoder_base_model_id", "Qwen/Qwen3-4B-Base")
    k_steps = config.get("model", {}).get("k_steps", 4)
    
    x_encoder = LatentEuclid(
        base_model_id=base_model_id,
        target_model_id=target_model_id,
        k_steps=k_steps
    ).to(device, dtype=torch.bfloat16)
    
    if os.path.exists(args.x_encoder_weights):
        print(f"Loading X-Encoder weights from {args.x_encoder_weights}...")
        state_dict = torch.load(args.x_encoder_weights, map_location="cpu", weights_only=False)
        x_encoder.load_state_dict(state_dict["model_state_dict"])
    else:
        print(f"Error: Could not find X-Encoder weights at {args.x_encoder_weights}")
        exit(1)
        
    x_encoder.eval()
    x_processor = x_encoder.processor
    
    # Load the Y-Encoder natively to embed the text options
    print(f"Loading native Y-Encoder ({decoder_base_model_id}) to embed text options...")
    y_tokenizer = AutoTokenizer.from_pretrained(decoder_base_model_id)
    y_encoder = AutoModelForCausalLM.from_pretrained(
        decoder_base_model_id, 
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None
    ).to(device)
    y_encoder.eval()
    
    with open("data/geothoughts_mcq.jsonl", 'r') as f:
        full_data = [json.loads(line) for line in f]
        
    with open("data/ground_truths_mcq.json", 'r') as f:
        ground_truths = json.load(f)
        
    # We do not need a train/test split. This is Zero-Shot evaluation over the entire manifold!
    dataset = GeoThoughtsMCQDataset(full_data, ground_truths, k_steps=k_steps)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=custom_collate)
    
    correct = 0
    incorrect = 0
    total = 0
    
    print("\n--- Starting Manifold Cosine Similarity Engine ---")
    
    for batch_idx, (images, texts, target_letters, option_maps) in enumerate(tqdm(loader)):
        batch_messages = []
        for img, txt in zip(images, texts):
            batch_messages.append([
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": txt}
                ]}
            ])
            
        rendered_texts = [x_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        inputs = x_processor(text=rendered_texts, images=images, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_latents = x_encoder(
                    input_ids=inputs.input_ids, 
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw")
                )
                
        # [Batch, 1024]
        thought_3 = predicted_latents[:, 3, :]
        
        for i in range(len(images)):
            t3_vector = thought_3[i] # [1024]
            ops = option_maps[i] # {'A': '10', 'B': '15', ...}
            
            best_sim = -float('inf')
            best_letter = None
            
            # Encode each option string through the target Y-Encoder
            for letter, opt_text in ops.items():
                opt_vector = get_y_encoder_embedding(y_encoder, y_tokenizer, opt_text, device) # [1024]
                
                # Compute Cosine Similarity between X-Encoder's Thought_3 and Y-Encoder's Text Option
                sim = F.cosine_similarity(t3_vector.unsqueeze(0), opt_vector.unsqueeze(0)).item()
                
                if sim > best_sim:
                    best_sim = sim
                    best_letter = letter
                    
            if best_letter == target_letters[i]:
                correct += 1
            else:
                incorrect += 1
                
            total += 1
            
        if (batch_idx + 1) % 10 == 0:
            ev_acc = ((correct * 1.0) - (incorrect * (1.0 / 3.0))) / total * 100.0
            print(f"\nBatch {batch_idx+1} | Acc: {correct}/{total} | Expected Value Score: {ev_acc:.2f}%")
            
    final_ev_acc = ((correct * 1.0) - (incorrect * (1.0 / 3.0))) / total * 100.0 if total > 0 else 0.0
    print("\n" + "="*50)
    print("FINAL MANIFOLD SIMILARITY RESULTS")
    print(f"Total Samples Tested: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Zero-Shot Expected Value Score: {final_ev_acc:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_evaluation()
