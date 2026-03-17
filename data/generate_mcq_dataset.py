import json
import random
import os

def generate_mcq_dataset(input_jsonl="data/geothoughts_verified.jsonl", gt_json="data/ground_truths.json", output_jsonl="data/geothoughts_mcq.jsonl", output_gt="data/ground_truths_mcq.json"):
    print(f"Loading dataset from {input_jsonl} and {gt_json}...")
    
    with open(input_jsonl, 'r') as f:
        data = [json.loads(line) for line in f]
        
    with open(gt_json, 'r') as f:
        ground_truths = json.load(f)
        
    # Collect all unique ground truth answers to use as distractors
    all_answers = list(set([str(v).strip() for v in ground_truths.values() if str(v).strip() != ""]))
    
    mcq_data = []
    mcq_gts = {}
    
    random.seed(42) # For reproducibility
    
    options_letters = ['A', 'B', 'C', 'D']
    
    for item in data:
        img_path = item["image_path"]
        correct_ans = str(ground_truths.get(img_path, "0")).strip()
        
        # Sample 3 random distractors that are not the correct answer
        distractors = random.sample([a for a in all_answers if a != correct_ans], 3)
        
        # Combine and shuffle
        options = distractors + [correct_ans]
        random.shuffle(options)
        
        # Find the correct letter
        correct_idx = options.index(correct_ans)
        correct_letter = options_letters[correct_idx]
        
        # Format the question to append the options
        original_q = item["question"]
        if "<image>" in original_q:
            base_q = original_q.replace("<image>", "").strip()
            mcq_q = f"{base_q}\nOptions:\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n<image>"
        else:
            mcq_q = f"{original_q}\nOptions:\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}"
            
        # Only retain what the Linear Probe needs
        new_item = {
            "image_path": img_path,
            "question": mcq_q
        }
        mcq_data.append(new_item)
        
        mcq_gts[img_path] = correct_letter
        
    print(f"Writing {len(mcq_data)} items to {output_jsonl}...")
    with open(output_jsonl, 'w') as f:
        for item in mcq_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Writing {len(mcq_gts)} ground truths to {output_gt}...")
    with open(output_gt, 'w') as f:
        json.dump(mcq_gts, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    generate_mcq_dataset()
