import json
import os
import numpy as np

def main():
    # 1. Load V4 Eval json
    with open("data/eval_v4_projection_and_unfrozen_layers.json", "r") as f:
        v4_eval = json.load(f)
        
    print(f"Loaded {len(v4_eval)} V4 evaluations.")
    
    # Build image_path to question mapping
    image_to_question = {}
    with open("data/geothoughts_verified.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            img_path_key = os.path.basename(item["image_path"])
            image_to_question[img_path_key] = item["question"]
            
    correct_lens = []
    failed_lens = []
    
    for item in v4_eval:
        raw_img_path = item["image"]
        is_correct = item["is_correct"]
        
        img_path_key = os.path.basename(raw_img_path)
        if img_path_key not in image_to_question:
            continue
            
        question = image_to_question[img_path_key]
        q_len = len(question.split())
        
        if is_correct:
            correct_lens.append(q_len)
        else:
            failed_lens.append(q_len)
            
    print("\n" + "="*50)
    print("=== Question Text Length Analysis (v4) ===")
    print("="*50)
    print(f"Correct Samples ({len(correct_lens)}):")
    print(f"  Avg Word Count: {np.mean(correct_lens):.1f} words")
    print(f"  Median Words:   {np.median(correct_lens):.1f} words")
    print(f"  Max Words:      {np.max(correct_lens)} words")
    
    print(f"\nFailed Samples ({len(failed_lens)}):")
    print(f"  Avg Word Count: {np.mean(failed_lens):.1f} words")
    print(f"  Median Words:   {np.median(failed_lens):.1f} words")
    print(f"  Max Words:      {np.max(failed_lens)} words")
    print("\nConclusion: If correct samples are significantly longer, the model may be ignoring vision and relying entirely on rich textual logic.")

if __name__ == "__main__":
    main()
