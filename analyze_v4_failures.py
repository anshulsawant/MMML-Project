import json
import random
import numpy as np

def main():
    # 1. Load Ground Truth Data
    ground_truths = {}
    with open("data/geothoughts_verified.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            ground_truths[data["image_path"]] = {
                "question": data["question"],
                "reasoning": data["reasoning"]
            }

    # 2. Load Evaluation Data
    with open("data/eval_v4_projection_and_unfrozen_layers.json", "r") as f:
        eval_data = json.load(f)

    correct_samples = []
    failed_samples = []

    for item in eval_data:
        image_path = item["image"]
        if image_path in ground_truths:
            merged = {
                "image": image_path,
                "question": ground_truths[image_path]["question"],
                "target_thought": ground_truths[image_path]["reasoning"],
                "is_correct": item["is_correct"],
                "gt_raw": item["gt_raw"],
                "pred_raw": item["pred_raw"],
                "gt_norm": item["gt_norm"],
                "pred_norm": item["pred_norm"],
                "model_generation": item["model_generation"]
            }
            if item["is_correct"]:
                correct_samples.append(merged)
            else:
                failed_samples.append(merged)

    print("=== V4 Failure Analysis ===")
    print(f"Total Samples: {len(eval_data)}")
    print(f"Correct: {len(correct_samples)}")
    print(f"Failed: {len(failed_samples)}")
    
    # Analyze by ground truth thought length (proxy for complexity/number of steps)
    correct_lengths = [len(s["target_thought"]) for s in correct_samples]
    failed_lengths = [len(s["target_thought"]) for s in failed_samples]
    
    print("\n--- Correlation: Target Thought Length (Complexity) vs Failure ---")
    print(f"Average Thought Length (Correct): {np.mean(correct_lengths):.2f} chars")
    print(f"Average Thought Length (Failed):  {np.mean(failed_lengths):.2f} chars")
    
    # Analyze empty generations
    empty_failed = [s for s in failed_samples if not str(s["model_generation"]).strip()]
    print(f"\n--- Generation Collapse ---")
    print(f"Failed samples with empty/whitespace generation: {len(empty_failed)} ({len(empty_failed)/len(failed_samples)*100:.2f}%)")
    
    # Analyze generation length
    correct_gen_lengths = [len(str(s["model_generation"])) for s in correct_samples]
    failed_gen_lengths = [len(str(s["model_generation"])) for s in failed_samples]
    print(f"Average Generation Length (Correct): {np.mean(correct_gen_lengths):.2f} chars")
    print(f"Average Generation Length (Failed):  {np.mean(failed_gen_lengths):.2f} chars")

    # Sample 100 Random Failures for Explorer
    random.seed(1337)
    sampled_failures = random.sample(failed_samples, min(100, len(failed_samples)))
    
    with open("data/v4_failures_explorer.json", "w") as f:
        json.dump(sampled_failures, f, indent=2)
        
    print(f"\nExported {len(sampled_failures)} random failed samples to data/v4_failures_explorer.json for UI analysis.")

if __name__ == "__main__":
    main()
