import os
import json
import base64
import time
from typing import List, Dict

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from ~/.env
load_dotenv(os.path.expanduser('~/.env'))

# Use the strictly typed Google GenAI Python SDK
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYS_PROMPT = """You are an expert geometry problem solver. Your task is to process geometry problems (Image + text prompt) and output a strict 4-step reasoning chain.

YOU MUST strictly follow this K=4 format exactly. Do not truncate your answer.
Step 1 [Visual Parsing]: Extract all explicit and implicit geometric relationships visible in the image (e.g., parallel lines, specific angle measures, lengths).
Step 2 [Theorem Retrieval]: State the formal mathematical theorems required to solve the problem based on the visual features (e.g., Alternate Interior Angles Theorem).
Step 3 [Calculation]: Perform the step-by-step arithmetic needed.
Step 4 [Final Conclusion]: Output a short string with the final numerical or logical answer (e.g., \\boxed{40}).

Do not deviate from 4 steps. Each step must be clearly numbered. Your response MUST contain all 4 steps."""

def extract_image_bytes(image_path: str) -> str:
    """Helper to load standard image bytes into base64."""
    with open(image_path, "rb") as int_img:
        return base64.b64encode(int_img.read()).decode('utf-8')

def build_inline_request(problem: Dict) -> Dict:
    """Constructs a GenerateContentRequest compatible format for exactly one problem."""
    path = problem["image_path"]
    mime_type = "image/jpeg" if path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    b64_string = extract_image_bytes(path)
    
    return {
        'contents': [{
            'parts': [
                {'text': f"{SYS_PROMPT}\n\nProblem:\n{problem['question']}"},
                {'inline_data': {'data': b64_string, 'mime_type': mime_type}}
            ],
            'role': 'user'
        }]
    }

import argparse

def main(dataset_path: str, output_path: str, images_dir: str, limit: int = None, target_sample: str = None):
    """Processes large JSONL datasets using the asynchronous genai Batch API."""
    
    print(f"Loading local dataset from {dataset_path}...")
    problems = []
    
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # GeoThought dataset typically provides an "image" field referring to the relative path
                image_rel_path = data.get("image", "")
                question_text = data.get("text", "")
                
                full_image_path = os.path.join(images_dir, image_rel_path)
                
                if target_sample and target_sample not in full_image_path:
                    continue
                
                if os.path.exists(full_image_path) and question_text:
                    problems.append({
                        "image_path": full_image_path,
                        "question": question_text
                    })
                    
                if limit and len(problems) >= limit:
                    break
    
    print(f"Loaded {len(problems)} pending inference problems.")
    
    if len(problems) == 0:
        print("No pending generation jobs found in the dataset list.")
        return
        
    print(f"Packaging {len(problems)} models into Inline Batch execution format...")
    inline_requests = [build_inline_request(p) for p in problems]
    
    print(f"Connecting to Gemini Batch API [Model: gemini-3-flash-preview]")
    inline_batch_job = client.batches.create(
        model="gemini-3-flash-preview",
        src=inline_requests,
        config={
            'display_name': "geothoughts-k4-flash-batch",
        },
    )
    
    job_name = inline_batch_job.name
    print(f"Created batch job: {job_name}")
    print(f"Polling status for background remote calculation...")

    while True:
        batch_job_inline = client.batches.get(name=job_name)
        if batch_job_inline.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'):
            break
        print(f"Job not finished. Current state: {batch_job_inline.state.name}. Waiting 30 seconds...")
        time.sleep(30)
        
    print(f"Job finished with state: {batch_job_inline.state.name}")
    
    if batch_job_inline.state.name == 'JOB_STATE_SUCCEEDED' and batch_job_inline.dest and batch_job_inline.dest.inlined_responses:
        valid_results = []
        # Ordered identically to src submission sequence natively by the API logic
        for i, inline_response in enumerate(batch_job_inline.dest.inlined_responses):
            p = problems[i]
            reasoning_out = ""
            
            if inline_response.response:
                reasoning_out = inline_response.response.text
                
            valid_results.append({
                "image_path": p["image_path"],
                "question": p["question"],
                "reasoning": reasoning_out.strip()
            })
            
            if (i + 1) % 50 == 0 or (i + 1) == len(inline_requests):
                print(f"Downloaded Sample {i+1} : {reasoning_out[:100]}...")
            
        with open(output_path, 'a') as f:
            for res in valid_results:
                f.write(json.dumps(res) + '\\n')
                
        print(f"Successfully processed {len(valid_results)} samples securely in batch mapping over to {output_path}!")
    else:
        print("Failed to map success state directly onto the inline return. Check server payload.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K=4 Euclidean Latent steps using Gemini")
    parser.add_argument("--dataset_path", type=str, default="GeoThought/playground/data/test_question.jsonl")
    parser.add_argument("--output_path", type=str, default="data/geothoughts_k4.jsonl")
    parser.add_argument("--images_dir", type=str, default="GeoThought/playground/data/")
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples to call API on")
    parser.add_argument("--target_sample", type=str, default=None, help="Specific sample to target (e.g., 'sample_13.jpg')")
    
    args = parser.parse_args()
    
    main(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        images_dir=args.images_dir,
        limit=args.limit,
        target_sample=args.target_sample
    )
