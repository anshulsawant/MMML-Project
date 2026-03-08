import argparse
import yaml
import json
import os
import torch
from PIL import Image
import random

from models.latent_euclid import LatentEuclid
from models.y_decoder_prefix import YDecoderPrefix

def generate_answers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/config_decoder.yaml")
    parser.add_argument("--decoder_weights", type=str, required=True, help="Path to decoder_epoch_X.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Loading X-Encoder...")
    x_encoder = LatentEuclid(
        base_model_id=config["model"]["base_model_id"],
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    ).to(device)
    
    # Load VICReg weights
    weight_path = config["model"].get("x_encoder_weights")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    x_encoder.load_state_dict(state_dict["model_state_dict"])
    x_encoder.eval()

    print("Loading Y-Decoder...")
    y_decoder = YDecoderPrefix(
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    ).to(device)
    
    # Load projection head weights
    decoder_state_dict = torch.load(args.decoder_weights, map_location="cpu")
    
    # Backward compatibility: Check if the saved checkpoint is from before the MLP upgrade
    if "weight" in decoder_state_dict and "0.weight" not in decoder_state_dict:
        import torch.nn as nn
        print("Detected old-style single Linear layer checkpoint. Downgrading architecture for evaluation...")
        y_decoder.prefix_projection = nn.Linear(y_decoder.embedding_dim, y_decoder.embedding_dim).to(torch.bfloat16).to(device)
        
    y_decoder.prefix_projection.load_state_dict(decoder_state_dict)
    y_decoder.eval()

    print("Loading Validation Dataset...")
    with open(config["data"]["jsonl_path"], 'r') as f:
        full_data = [json.loads(line) for line in f]
        
    with open("data/ground_truths.json", 'r') as f:
        ground_truths = json.load(f)

    random.seed(42)
    random.shuffle(full_data)
    split_idx = int(0.9 * len(full_data))
    val_data = full_data[split_idx:]
    
    # Take 5 random validation samples to inspect
    samples = random.sample(val_data, 5)

    print("\n================ EVALUATION 🤖 ================\n")
    
    with torch.no_grad():
        for item in samples:
            img_path = item["image_path"]
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue
                
            question = item["question"]
            true_answer = ground_truths.get(img_path, "UNKNOWN")
            
            # Format X-encoder input
            thought_string = "".join([f"<thought_{i+1}>" for i in range(config["model"]["k_steps"])])
            full_text = question + " " + thought_string
            inputs = x_encoder.tokenizer([full_text], return_tensors="pt", padding=True).to(device)
            
            # 1. Get Geometric Continuous vectors [1, K, 4096]
            predicted_latents = x_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
            # 2. Generate Discrete Text [String]
            # y_decoder_prefix.generate() assumes empty prompt unless specified
            generated_text = y_decoder.generate(predicted_latents=predicted_latents, max_new_tokens=15)[0]
            
            print(f"Image: {img_path}")
            print(f"Question: {question.split('<image>')[0][:100]}...")
            print(f"Ground Truth : {true_answer}")
            print(f"Model Gen    : {generated_text.strip()}")
            print("-" * 50)

if __name__ == "__main__":
    generate_answers()
