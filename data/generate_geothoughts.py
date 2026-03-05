import os
import json
import base64
import argparse
import requests
import re
from typing import Dict, List

from google import genai
from dotenv import load_dotenv

load_dotenv(os.path.expanduser('~/.env'))
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYS_PROMPT = """You are an expert geometry problem solver. Your task is to process geometry problems (Image + text prompt) and output a strict 4-step reasoning chain.

YOU MUST strictly follow this K=4 format exactly. Do not truncate your answer.
Step 1 [Visual Parsing]: Extract all explicit and implicit geometric relationships visible in the image (e.g., parallel lines, specific angle measures, lengths).
Step 2 [Theorem Retrieval]: State the formal mathematical theorems required to solve the problem based on the visual features (e.g., Alternate Interior Angles Theorem).
Step 3 [Calculation]: Perform the step-by-step arithmetic needed.
Step 4 [Final Conclusion]: Output a short string with the final numerical or logical answer (e.g., \\boxed{40}).

Do not deviate from 4 steps. Each step must be clearly numbered. Your response MUST contain all 4 steps."""

def extract_image_bytes(image_path: str) -> str:
    with open(image_path, "rb") as int_img:
        return base64.b64encode(int_img.read()).decode('utf-8')

def build_batch_request_line(problem: Dict) -> str:
    path = problem["image_path"]
    mime_type = "image/jpeg" if path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    b64_string = extract_image_bytes(path)
    
    match = re.search(r'sample_(\d+)\.', path)
    problem_idx = match.group(1) if match else "0"
    
    payload = {
        "custom_job_id": str(problem_idx),
        "request": {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": f"{SYS_PROMPT}\n\nProblem:\n{problem['question']}"},
                    {"inline_data": {"data": b64_string, "mime_type": mime_type}}
                ]
            }]
        }
    }
    return json.dumps(payload)

def cmd_submit(dataset_path: str, images_dir: str, limit: int = None, target_sample: str = None):
    print(f"Loading local dataset from {dataset_path}...")
    problems = []
    
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_rel_path = data.get("image", "")
                question_text = data.get("text", "")
                full_image_path = os.path.join(images_dir, image_rel_path)
                
                if target_sample and target_sample not in full_image_path:
                    continue
                
                if os.path.exists(full_image_path) and question_text:
                    problems.append({"image_path": full_image_path, "question": question_text})
                    
                if limit and len(problems) >= limit:
                    break

    if not problems:
        print("No pending generation jobs found in the dataset list.")
        return

    tmp_jsonl = "batch_requests.jsonl"
    print(f"Packaging {len(problems)} models into JSONL Batch execution format at {tmp_jsonl}...")
    with open(tmp_jsonl, 'w') as f:
        for p in problems:
            f.write(build_batch_request_line(p) + '\n')
            
    print(f"Uploading {tmp_jsonl} to Google GenAI File API...")
    upload_file = client.files.upload(file=tmp_jsonl, config={'mime_type': 'application/jsonl'})
    print(f"File uploaded. URI: {upload_file.uri}, Name: {upload_file.name}")
    
    print("Connecting to Gemini Batch API [Model: gemini-3.1-pro-preview]")
    batch_job = client.batches.create(
        model="gemini-3.1-pro-preview",
        src=upload_file.name,
        config={'display_name': "geothoughts-k4-3.1-pro"},
    )
    
    # Save the job ID locally so it's not lost
    job_id_file = "data/latest_gemini_job_id.txt"
    with open(job_id_file, 'w') as f:
        f.write(batch_job.name)
        
    print(f"\n+++ BATCH SUBMITTED +++")
    print(f"Batch Name: {batch_job.name}")
    print(f"Saved locally to: {job_id_file}")
    print(f"State: {batch_job.state.name}")
    print(f"\nRun this script again whenever you want to check status/download:")
    print(f"python data/generate_geothoughts.py --mode retrieve")

def cmd_retrieve(job_name: str, output_path: str, dataset_path: str, images_dir: str):
    print(f"Polling status for background job: {job_name} ...")
    batch_job = client.batches.get(name=job_name)
    state = batch_job.state.name
    print(f"Current State: {state}")
    
    if state == 'JOB_STATE_SUCCEEDED':
        print(f"Job completed successfully. Downloading results from {batch_job.dest.uri}...")
        
        response = requests.get(batch_job.dest.uri)
        if response.status_code != 200:
            print(f"Failed to download from {batch_job.dest.uri}. HTTP {response.status_code}")
            return
            
        output_lines = response.text.strip().split('\n')
        print(f"Downloaded {len(output_lines)} lines.")
        
        questions_map = {}
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    match = re.search(r'sample_(\d+)\.', data.get("image", ""))
                    if match:
                        questions_map[str(match.group(1))] = data.get("text", "")

        valid_results = []
        for line in output_lines:
            if not line.strip(): continue
            parsed = json.loads(line)
            
            idx = parsed.get("custom_job_id")
            if not idx: continue
            
            error = parsed.get("error")
            if error:
                print(f"Warning: Item {idx} failed on server: {error}")
                continue
                
            resp = parsed.get("response")
            reasoning_out = ""
            if resp and "candidates" in resp and len(resp["candidates"]) > 0:
                parts = resp["candidates"][0]["content"]["parts"]
                reasoning_out = parts[0]["text"]
            
            image_path = os.path.join(images_dir, "images", "samples", f"sample_{idx}.jpg")
            
            q = questions_map.get(str(idx), "")
            valid_results.append({
                "image_path": image_path,
                "question": q,
                "reasoning": reasoning_out.strip()
            })
                        
        with open(output_path, 'a') as f:
            for res in valid_results:
                f.write(json.dumps(res) + '\n')
                
        print(f"Successfully saved {len(valid_results)} aligned samples explicitly linked by custom_job_id permanently to {output_path}!")

    else:
        print("Job is not finished yet or failed. Check back later.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K=4 Euclidean Latent steps using Gemini")
    parser.add_argument("--mode", type=str, choices=['submit', 'retrieve'], required=True, help="Submit a new batch or retrieve a finished one.")
    parser.add_argument("--dataset_path", type=str, default="GeoThought/playground/data/test_question.jsonl")
    parser.add_argument("--output_path", type=str, default="data/geothoughts_k4_gemini3.1.jsonl")
    parser.add_argument("--images_dir", type=str, default="GeoThought/playground/data/")
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples to call API on")
    parser.add_argument("--target_sample", type=str, default=None, help="Specific sample to target")
    parser.add_argument("--job_name", type=str, default=None, help="Job name required for retrieving (e.g., batches/12345)")
    
    args = parser.parse_args()
    
    if args.mode == "submit":
        cmd_submit(args.dataset_path, args.images_dir, args.limit, args.target_sample)
    elif args.mode == "retrieve":
        # If no job_name provided, try to load it from the tracker file
        job_id = args.job_name
        tracker_file = "data/latest_gemini_job_id.txt"
        
        if not job_id:
            if os.path.exists(tracker_file):
                with open(tracker_file, 'r') as f:
                    job_id = f.read().strip()
                print(f"Auto-loaded job_name '{job_id}' from {tracker_file}")
            else:
                print("Error: --job_name is required for retrieve mode, and no tracking file was found.")
                
        if job_id:
            cmd_retrieve(job_id, args.output_path, args.dataset_path, args.images_dir)
