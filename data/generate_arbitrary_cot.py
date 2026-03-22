import os
import json
import argparse
import time
from tqdm import tqdm
from PIL import Image
from typing import Dict, List

from google import genai
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore', message='.*there are non-text parts in the response.*')

load_dotenv(os.path.expanduser('~/.env'))
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYS_PROMPT = """You are an expert geometry problem solver. Your task is to process geometry problems (Image + text prompt) and output a detailed, arbitrary-length step-by-step reasoning chain.

YOU MUST strictly follow this formatting constraint:
- Break your reasoning down into as many logical steps as necessary to solve the problem mathematically.
- Every single logical step MUST start on a new line and begin explicitly with exactly "Step 1:", "Step 2:", etc.
- Your VERY LAST STEP must begin with exactly "Step N [Final Conclusion]:" (where N is your final step number) and contain a short string with the final numerical or logical answer (e.g., \\boxed{40}).

Example Output Format:
Step 1: Parse the visible geometric relationships...
Step 2: Recall the Alternate Interior Angles Theorem...
Step 3: Calculate the interior sum...
Step 4 [Final Conclusion]: \boxed{40}

Do not combine steps. Number them sequentially and cleanly."""

def call_gemini_flash(image_path: str, question: str) -> str:
    image = Image.open(image_path)
    
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[
            f"{SYS_PROMPT}\n\nProblem:\n{question}",
            image
        ],
        config=genai.types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        )
    )
    return response.text

def process_dataset(dataset_path: str, images_dir: str, output_path: str, limit: int = None, target_sample: str = None):
    processed_samples = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    processed_samples.add(data.get("image_path", ""))
                except json.JSONDecodeError:
                    pass
    
    print(f"Skipping {len(processed_samples)} previously processed items from {output_path}.")
    
    problems = []
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_rel_path = data.get("image", "")
                question = data.get("text", "")
                full_image_path = os.path.join(images_dir, image_rel_path)
                
                if target_sample and target_sample not in full_image_path:
                    continue
                    
                if full_image_path in processed_samples:
                    continue
                    
                if os.path.exists(full_image_path) and question:
                    problems.append({"image_path": full_image_path, "question": question})

    print(f"Loaded {len(problems)} new problems to run through gemini-3.0-flash.")
    
    if limit:
        problems = problems[:limit]
        print(f"Limited run to {limit} items.")

    success_count = 0
    import concurrent.futures
    import threading
    import logging

    # Suppress all internal SDK prints and warnings that ruin TQDM
    logging.getLogger("google").setLevel(logging.ERROR)

    api_lock = threading.Lock()
    file_lock = threading.Lock()
    last_req_time = 0.0

    def process_item(p):
        nonlocal success_count, last_req_time
        
        # Enforce global 0.45s delay between API hits (max ~133 RPM)
        with api_lock:
            now = time.time()
            elapsed = now - last_req_time
            if elapsed < 0.45:
                time.sleep(0.45 - elapsed)
            last_req_time = time.time()
            
        try:
            reasoning = call_gemini_flash(p['image_path'], p['question'])
            
            result = {
                "image_path": p['image_path'],
                "question": p['question'],
                "reasoning": reasoning.strip()
            }
            print(f"\n======================\n[Question]: {p['question']}\n\n[Reasoning]:\n{reasoning.strip()}\n======================")
            with file_lock:
                with open(output_path, 'a') as f:
                    f.write(json.dumps(result) + '\n')
                success_count += 1
        except Exception as e:
            # Swallow timeouts/errors silently so TQDM keeps drawing smoothly. Next run will catch them.
            time.sleep(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_item, p): p for p in problems}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(problems), desc="Generating Arbitrary Reasoning"):
            pass

    print(f"\\n+++ FINISHED! +++")
    print(f"Successfully generated {success_count} geometries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arbitrary length Latent logic steps using Gemini 3.0 Flash")
    parser.add_argument("--dataset_path", type=str, default="GeoThought/playground/data/test_question.jsonl")
    parser.add_argument("--output_path", type=str, default="data/geothoughts_arbitrary_cot.jsonl")
    parser.add_argument("--images_dir", type=str, default="GeoThought/playground/data/")
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples to call API on")
    parser.add_argument("--target_sample", type=str, default=None, help="Specific sample to target")
    
    args = parser.parse_args()
    process_dataset(args.dataset_path, args.images_dir, args.output_path, args.limit, args.target_sample)
