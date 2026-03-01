import torch
import torch.nn as nn
from transformers import AutoConfig

# Import our custom modules
from models.latent_euclid import LatentEuclid
from training.stable_alignment_loss import AlignmentLossFactory

def run_cpu_smoke_test():
    print("Beginning Rapid CPU Smoke Test with Tiny Qwen3-VL...")
    
    # Tiny model for CI/CD CPU testing
    tiny_base_model = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
    # For testing purposes, we can use the same tiny model to get a small target dimension config
    tiny_target_model = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
    k_steps = 4
    batch_size = 2
    
    print(f"Loading LatentEuclid with base_model={tiny_base_model}")
    
    try:
        model = LatentEuclid(
            base_model_id=tiny_base_model,
            target_model_id=tiny_target_model,
            k_steps=k_steps
        )
        # Move to CPU for smoke testing
        model = model.to('cpu')
        
    except Exception as e:
        print(f"Failed to load tiny model: {e}")
        return
        
    print("\n--- Testing LatentEuclid Forward Pass ---")
    
    # Create fake inputs. The tokenizer adds <thought> tokens.
    text_prompts = [
        "Find the value of x in the triangle. <thought_1><thought_2><thought_3><thought_4>",
        "Find the value of x in the triangle. <thought_1><thought_2><thought_3><thought_4>"
    ]
    
    device = "cpu"
    # Tokenize (we must make sure the <thought> tokens are in the input)
    # Alternatively, we just use the tokenizer we built in LatentEuclid
    inputs = model.tokenizer(text_prompts, return_tensors="pt").to(device)
    
    # Fake pixel values if the vision model requires them (typically a generic shape, e.g., [batch, channels, time, h, w] or similar)
    # We will just try to run text-only forward if the model supports it, or pass dummy pixels.
    # Qwen-VL typically needs `pixel_values` and `image_grid_thw` if images are specified,
    # but for text-only it might bypass vision encoding. Let's test text-only route first 
    # to test the causal masking and sequence extraction logic.
    
    try:
        predicted_latents = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        
        target_dim = model.predictor.mlp[-1].out_features
        print(f"Expected Shape: [{batch_size}, {k_steps}, {target_dim}]")
        print(f"Actual Shape:   {list(predicted_latents.shape)}")
        assert predicted_latents.shape == (batch_size, k_steps, target_dim), "Shape mismatch in LatentEuclid Output"
        
        print("\n--- Testing AlignmentLossFactory (VICReg) ---")
        dummy_targets = torch.randn(batch_size, k_steps, target_dim, device=device, dtype=predicted_latents.dtype)
        loss_factory_vicreg = AlignmentLossFactory(loss_type="vicreg")
        loss_vicreg = loss_factory_vicreg(predicted_latents, dummy_targets)
        print(f"VICReg Loss: {loss_vicreg.item():.4f}")
        assert not torch.isnan(loss_vicreg), "VICReg Loss returned NaN!"
        
        print("\n--- Testing Gradients (Backward Pass) ---")
        loss_vicreg.backward()
        print("Backward pass executed successfully. Computational graph is connected.")
        
        print("\n✅ All CPU Smoke Tests Passed Successfully!")
        
    except Exception as e:
        print(f"Error during forward/backward pass: {e}")

if __name__ == "__main__":
    run_cpu_smoke_test()
