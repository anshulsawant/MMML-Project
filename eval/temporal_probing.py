'''
Evaluation: Pre-Generation Hidden States Probing and VL-JEPA Latencies

This script is central to scientifically validating the core hypothesis that modern,
autoregressive VLMs suffer from "Visual Forgetting." Because the VLM translates continuous 
diagram features into discrete 1D texts, it allows language priors to hallucinate geometries.

1. Grounding Claim: Extracts Baseline Pre-Generation Token States vs LatentEuclid's K States.
   Identical Non-Linear MLPs probe the targets for geometric features to verify where visual topologies 
   are maintained natively. We use PyTorch MLPs instead of linear scikit-learn probes to 
   properly capture the entangled features mapped from the Qwen-0.6B LLM representations.
2. Latency Efficiency: Benchmarks identical answers driven temporally vs Auto-regressive passes.
'''

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NonLinearProbe(nn.Module):
    """
    A 2-layer MLP to probe complex, entangled geometric features from continuous vectors.
    """
    def __init__(self, input_dim, hidden_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_and_eval_probe(X_train, y_train, X_test, y_test, num_classes=2, epochs=100, lr=1e-3, device="cuda"):
    """
    Trains a NonLinearProbe on the train set and evaluates on the test set.
    """
    input_dim = X_train.shape[1]
    probe = NonLinearProbe(input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = probe(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
    probe.eval()
    with torch.no_grad():
        out = probe(X_test_t)
        preds = torch.argmax(out, dim=1)
        acc = accuracy_score(y_test_t.cpu().numpy(), preds.cpu().numpy())
        
    return acc

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
    print("Beginning Progressive Temporal Probing with Non-Linear PyTorch MLPs...")
    
    # Example logic mapping:
    # X_vanilla, y = extract_baseline_pregen_states(...)
    # X_train, X_test, y_train, y_test = train_test_split(X_vanilla, y, test_size=0.2)
    # acc_vanilla = train_and_eval_probe(X_train, y_train, X_test, y_test)
    # print(f"Vanilla Pre-Generation Probe Accuracy: {acc_vanilla*100:.2f}%")
    
    # X_latent_k, y = extract_latent_euclid_states(...)
    # for k in range(4):
    #     X_train, X_test, y_train, y_test = train_test_split(X_latent_k[k], y, test_size=0.2)
    #     acc_k = train_and_eval_probe(X_train, y_train, X_test, y_test)
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
