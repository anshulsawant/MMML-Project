import torch
import numpy as np
import warnings

from eval.temporal_probing import extract_baseline_pregen_states, extract_latent_euclid_states

class MockModel(torch.nn.Module):
    def __init__(self, target_dim=16, k_steps=4, is_vanilla=False):
        super().__init__()
        self.target_dim = target_dim
        self.k_steps = k_steps
        self.is_vanilla = is_vanilla
        
    def eval(self):
        pass
        
    def forward(self, input_ids, pixel_values=None, output_hidden_states=False):
        batch_size = input_ids.shape[0]
        
        # Mock Vanilla Return Signature
        if self.is_vanilla:
            # outputs.hidden_states[-1]
            mock_hidden = torch.randn(batch_size, 50, self.target_dim) # [batch, seq_len, dim]
            class MockOutput:
                hidden_states = [mock_hidden]
            return MockOutput()
            
        # Mock LatentEuclid Return Signature
        return torch.randn(batch_size, self.k_steps, self.target_dim) # [batch, K, dim]


def test_probing_extraction():
    print("\n--- Testing Evaluation Probe Extractions ---")
    
    batch_size = 4
    target_dim = 16
    k_steps = 4
    
    # Create Mock Dataloader Output
    mock_batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, 50)),
        "pixels": torch.randn(batch_size, 3, 224, 224),
        "probe_labels": [1, 0, 1, 0]
    }
    
    mock_dataloader = [mock_batch]
    
    # 1. Test Vanilla Extraction
    print("Testing Vanilla Pre-Generation State Extraction...")
    vanilla_model = MockModel(target_dim=target_dim, is_vanilla=True)
    
    vanilla_states, vanilla_labels = extract_baseline_pregen_states(vanilla_model, mock_dataloader, device="cpu")
    
    print(f"Vanilla States Shape: {vanilla_states.shape}")
    assert vanilla_states.shape == (batch_size, target_dim), f"Vanilla extraction shape mismatch. Got {vanilla_states.shape}"
    assert vanilla_labels.shape == (batch_size,), "Vanilla labels shape mismatch."
    
    # 2. Test LatentEuclid Extraction
    print("\nTesting LatentEuclid K-Thought State Extraction...")
    latent_model = MockModel(target_dim=target_dim, k_steps=k_steps, is_vanilla=False)
    
    latent_states_dict, latent_labels = extract_latent_euclid_states(latent_model, mock_dataloader, device="cpu")
    
    print(f"LatentEuclid States Keys: {list(latent_states_dict.keys())}")
    assert len(latent_states_dict) == k_steps, f"Expected {k_steps} dictionaries, got {len(latent_states_dict)}"
    
    for k in range(k_steps):
        state_k = latent_states_dict[k]
        assert state_k.shape == (batch_size, target_dim), f"Thought {k} shape mismatch. Got {state_k.shape}"
        
    assert latent_labels.shape == (batch_size,), "LatentEuclid labels shape mismatch."
    
    print("\n✅ All Evaluation Extraction Pipelines Passed!")


def run_all_eval_smoke_tests():
    print("Beginning LatentEuclid Probing Evaluation Smoke Tests...")
    test_probing_extraction()
    print("\n✅ All Eval Smoke Tests Finished!")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_all_eval_smoke_tests()
