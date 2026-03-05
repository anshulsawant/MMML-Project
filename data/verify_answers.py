import os
import json
import re
from datasets import load_dataset
from tqdm import tqdm

def extract_gemini_answer(reasoning: str) -> str:
    """Extracts the value inside \boxed{} from Gemini's reasoning."""
    match = re.search(r'\\boxed{([^>]*?)}', reasoning)
    if match:
        return match.group(1).strip()
    return None

def extract_ground_truth(solution: str) -> str:
    """Extracts the Final Answer from the dataset's solution."""
    match = re.search(r'Final Answer:\s*(.*?)\s*</answer>', solution, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def normalize_answer(ans: str) -> str:
    """Normalizes answers for comparison (e.g., stripping degrees, spaces, basic equivalents)."""
    if ans is None:
        return ""
    
    ans = ans.lower()
    # Remove degree symbols or text
    ans = ans.replace('^\circ', '').replace('^\\circ', '').replace('\circ', '').replace('°', '')
    ans = ans.replace(' degrees', '').replace(' degree', '')
    
    ans = ans.strip()
    
    # Optional: try to convert to float to match 20.0 with 20
    try:
        val = float(ans)
        # return string representation of float without trailing .0
        if val.is_integer():
            return str(int(val))
        return str(val)
    except ValueError:
        return ans

def main():
    jsonl_path = "data/geothoughts_k4_gemini3.1.jsonl"
    parquet_path = "GeoThought/playground/data/geo_thought/Geo-Thought-6K.parquet"
    
    print("Loading Ground Truth Dataset...")
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    
    correct = 0
    total = 0
    missing_gemini = 0
    missing_gt = 0
    
    print("Evaluating Gemini Flash Outputs...")
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            
            # Extract index from 'GeoThought/playground/data/images/samples/sample_idx.jpg'
            match = re.search(r'sample_(\d+)\.jpg', data['image_path'])
            if not match:
                continue
            idx = int(match.group(1))
            
            # Retrieve ground truth
            gt_row = dataset[idx]
            gt_answer_raw = extract_ground_truth(gt_row['solution'])
            
            # Retrieve Gemini answer
            gemini_answer_raw = extract_gemini_answer(data.get('reasoning', ''))
            
            gt_answer = normalize_answer(gt_answer_raw)
            gemini_answer = normalize_answer(gemini_answer_raw)
            
            if not gemini_answer_raw:
                missing_gemini += 1
            if not gt_answer_raw:
                missing_gt += 1
                continue # If we can't parse GT, skip
                
            total += 1
            if gt_answer == gemini_answer:
                correct += 1
            else:
                # Optionally print failures for debugging
                if total <= 10: # Print first 10 mistakes
                    print(f"Mismatch [{idx}]: GT='{gt_answer_raw}' -> '{gt_answer}', Gemini='{gemini_answer_raw}' -> '{gemini_answer}'")
                    
    print("\n--- Evaluation Results ---")
    print(f"Total Valid Samples: {total}")
    print(f"Correct Matches: {correct}")
    print(f"Accuracy: {(correct / total * 100) if total > 0 else 0:.2f}%")
    print(f"Failed to parse Gemini box: {missing_gemini}")
    print(f"Failed to parse Ground Truth: {missing_gt}")

if __name__ == "__main__":
    main()
