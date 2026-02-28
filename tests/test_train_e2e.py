import os
import sys
import unittest.mock
import warnings
import traceback

from training.train_x_encoder import train

def run_end_to_end_smoke_test():
    print("\n--- Testing End-to-End LatentEuclid SFT Training Loop ---")
    print("Using 1 CPU Node, Tiny-Qwen3 Models, Batch Size 2, 1 Epoch...")
    
    config_path = "tests/mock_data/test_config.yaml"
    out_file = "latent_euclid_x_encoder_final.pt"
    
    # Clean up any old output states
    if os.path.exists(out_file):
        os.remove(out_file)
        
    # Patch the argparse inputs directly to bypass CLI
    test_args = ["train_x_encoder.py", "--config", config_path]
    
    with unittest.mock.patch.object(sys, 'argv', test_args):
        try:
            train()
            assert os.path.exists(out_file), "Expected model state dict .pt was not saved."
            print("\n✅ End-to-End LatentEuclid SFT Training Loop test passed.")
        except Exception as e:
            print("\n❌ End-to-End Training Loop failed!")
            traceback.print_exc()
            raise e
        finally:
            # Clean up the artifact so it doesn't pollute the repo
            if os.path.exists(out_file):
                os.remove(out_file)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_end_to_end_smoke_test()
