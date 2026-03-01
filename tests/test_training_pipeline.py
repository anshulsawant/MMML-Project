import os
import json
import yaml
import torch
import warnings
from PIL import Image

# Import functions/classes to test
from training.train_x_encoder import GeoThoughtsDataset, custom_collate

def test_yaml_config():
    print("\n--- Testing YAML Config Parsing ---")
    config_path = "training/config.yaml"
    assert os.path.exists(config_path), "config.yaml not found!"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    assert "model" in config, "model section missing from config"
    assert "training" in config, "training section missing from config"
    assert "data" in config, "data section missing from config"
    assert config["model"]["k_steps"] == 4, "k_steps should be 4"
    assert config["training"]["loss_type"] in ["info_nce_vanilla", "info_nce_threshold", "vicreg"], "Invalid loss type in config"
    
    print("✅ YAML Config parses cleanly and contains expected schema.")

def test_geothoughts_dataset_and_collate():
    print("\n--- Testing GeoThoughtsDataset & custom_collate ---")
    
    # 1. Setup mock data
    test_dir = "tests/mock_data"
    targets_dir = os.path.join(test_dir, "target_tensors")
    os.makedirs(targets_dir, exist_ok=True)
    
    jsonl_path = os.path.join(test_dir, "mock_geothoughts.jsonl")
    
    # Create mock target tensor [K, target_dim]
    dummy_tensor = torch.randn(4, 16)
    torch.save(dummy_tensor, os.path.join(targets_dir, "problem_0_targets.pt"))
    torch.save(dummy_tensor, os.path.join(targets_dir, "problem_1_targets.pt"))
    
    # Create mock JSONL
    mock_data = [
        {"image_path": "invalid_path.jpg", "question": "What is x?"},
        {"image_path": "another_invalid.png", "question": "Find the angle."}
    ]
    
    with open(jsonl_path, "w") as f:
        for item in mock_data:
            f.write(json.dumps(item) + "\n")
            
    # 2. Test Dataset Instantiation (No real tokenizer needed for this test, we can pass None since it's not used in __getitem__)
    dataset = GeoThoughtsDataset(
        jsonl_path=jsonl_path,
        targets_dir=targets_dir,
        tokenizer=None,
        k_steps=4
    )
    
    assert len(dataset) == 2, f"Expected 2 items in dataset, got {len(dataset)}"
    
    # 3. Test __getitem__
    item0 = dataset[0]
    assert isinstance(item0["image"], Image.Image), "Image failed to load or fallback"
    assert "<thought_1><thought_2><thought_3><thought_4>" in item0["text"], "Thought string not appended correctly"
    assert "What is x?" in item0["text"], "Original question missing from output text"
    assert item0["target"].shape == (4, 16), "Target tensor shape mismatch"
    
    # 4. Test custom_collate
    batch = [dataset[0], dataset[1]]
    images, texts, targets = custom_collate(batch)
    
    assert len(images) == 2, "Batch should have 2 images"
    assert len(texts) == 2, "Batch should have 2 text strings"
    assert targets.shape == (2, 4, 16), "Batched target tensor should be [batch_size, k_steps, target_dim]"
    
    print("✅ Dataset and Collation logic tests passed.")

def run_all_smoke_tests():
    print("Beginning Training Pipeline Smoke Tests...")
    test_yaml_config()
    test_geothoughts_dataset_and_collate()
    print("\n✅ All Training Pipeline Smoke Tests Finished!")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_all_smoke_tests()
