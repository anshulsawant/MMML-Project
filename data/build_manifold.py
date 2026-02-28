import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re


MODEL_ID = "Qwen/Qwen3-0.6B" # Using 0.6B target manifold mode

def load_qwen_target_model():
    print(f"Loading {MODEL_ID} for Target Manifold extraction...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model

def parse_k4_steps(reasoning_text: str):
    """Extracts exactly 4 steps from the LLM output."""
    steps = []
    # Simple regex to split based on "Step X:" formatting constraint
    parts = re.split(r'Step \d+.*?\]?:', reasoning_text)
    for p in parts[1:]: # Skip text before Step 1
        steps.append(p.strip())
        
    # Validation padding
    while len(steps) < 4:
        steps.append("Empty Step")
    return steps[:4]

def embed_step(text: str, tokenizer, model, device="cuda"):
    """Passes text natively through Qwen3-0.6B and extracts the final hidden state."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Extract the hidden states from the last layer [batch, seq_len, hidden_dim]
        last_hidden_states = outputs.hidden_states[-1]
        
        # Take the embedding of the final token as the sequence representation
        # Shape: [1, hidden_dim]
        final_token_embedding = last_hidden_states[:, -1, :] 
    
    return final_token_embedding.squeeze(0).cpu()

def build_manifold(input_jsonl: str, output_dir: str):
    """Processes K4 text and saves continuous target tensors."""
    tokenizer, model = load_qwen_target_model()
    
    with open(input_jsonl, 'r') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            steps = parse_k4_steps(data["reasoning"])
            
            problem_tensors = []
            for step_text in steps:
                vec = embed_step(step_text, tokenizer, model)
                problem_tensors.append(vec)
            
            # Stack into [4, hidden_dim]
            target_tensor = torch.stack(problem_tensors)
            
            # Save the .pt file for Phase 3 training
            torch.save(target_tensor, f"{output_dir}/problem_{idx}_targets.pt")

if __name__ == "__main__":
    # Example usage:
    # build_manifold("geothoughts_k4.jsonl", "target_tensors/")
    pass
