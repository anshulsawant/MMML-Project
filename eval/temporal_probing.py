import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def extract_baseline_pregen_states(vanilla_model, dataloader, device="cuda"):
    """
    Passes image + question into Vanilla Qwen3-VL and extracts the final hidden state 
    of the sequence right BEFORE autoregressive text generation begins.
    """
    vanilla_model.eval()
    vanilla_model.to(device)
    
    pregen_vectors = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            pixels = batch["pixels"].to(device)
            
            # Forward pass without generation
            outputs = vanilla_model(
                input_ids=inputs, 
                pixel_values=pixels,
                output_hidden_states=True
            )
            
            # Extract final layer, final sequence token
            # Shape: [batch, hidden_dim]
            last_hidden_states = outputs.hidden_states[-1]
            pregen_states = last_hidden_states[:, -1, :].cpu().numpy()
            
            pregen_vectors.extend(pregen_states)
            labels.extend(batch["probe_labels"]) # e.g. "Is AB parallel to CD?" (0 or 1)
            
    return np.array(pregen_vectors), np.array(labels)

def extract_latent_euclid_states(latent_model, dataloader, device="cuda"):
    """
    Extracts the K thought vectors predicted by LatentEuclid in a single forward pass.
    """
    latent_model.eval()
    latent_model.to(device)
    
    # Store K separate lists for progressive probing
    k_vectors = {i: [] for i in range(latent_model.k_steps)}
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            pixels = batch["pixels"].to(device)
            
            # Get the [batch, K, target_dim] latents
            predicted_targets = latent_model(
                input_ids=inputs, 
                pixel_values=pixels
            )
            
            for k in range(latent_model.k_steps):
                # Shape: [batch, target_dim]
                step_k_vectors = predicted_targets[:, k, :].cpu().numpy()
                k_vectors[k].extend(step_k_vectors)
                
            labels.extend(batch["probe_labels"])
            
    return {k: np.array(v) for k, v in k_vectors.items()}, np.array(labels)

def temporal_probing_experiment():
    """
    The Master Probing claim ensuring the model isn't Suffering from Visual Forgetting.
    If Vanilla fails but LatentEuclid Phase 1 (<thought_1>) succeeds, 
    the VL-JEPA hypothesis is fully validated.
    """
    print("Beginning Progressive Temporal Probing...")
    
    # 1. Standard train/test split on extracted vectors 
    # (Placeholder arrays for the scaffold)
    # X_vanilla, y = extract_baseline_pregen_states(...)
    # X_latent_k, y = extract_latent_euclid_states(...)
    
    # Example logic:
    # clf_vanilla = LogisticRegression(max_iter=1000).fit(X_vanilla_train, y_train)
    # acc_vanilla = accuracy_score(y_test, clf_vanilla.predict(X_vanilla_test))
    # print(f"Vanilla Pre-Generation Probe Accuracy: {acc_vanilla*100:.2f}%")
    
    # for k in range(4):
    #     clf_k = LogisticRegression(max_iter=1000).fit(X_latent_k_train[k], y_train)
    #     acc_k = accuracy_score(y_test, clf_k.predict(X_latent_k_test[k]))
    #     print(f"LatentEuclid <Thought_{k+1}> Probe Accuracy: {acc_k*100:.2f}%")

def measure_latency_win():
    """
    Placeholder wrapper to benchmark the single-shot LatentEuclid pass 
    + tiny decoder generating `target` text vs the massive autoregressive vanilla model.
    """
    print("Measuring Latency (Time-to-Answer) and FLOPs...")
    pass

if __name__ == "__main__":
    temporal_probing_experiment()
    measure_latency_win()
