import os
import json
import asyncio
from typing import List, Dict

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from ~/.env
load_dotenv(os.path.expanduser('~/.env'))

# Gemini acts as an OpenAI-compatible endpoint now
client = AsyncOpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYS_PROMPT = """You are an expert geometry problem solver. Your task is to process geometry problems (Image + text prompt) and output a strict 4-step reasoning chain.

YOU MUST strictly follow this K=4 format exactly. Do not truncate your answer.
Step 1 [Visual Parsing]: Extract all explicit and implicit geometric relationships visible in the image (e.g., parallel lines, specific angle measures, lengths).
Step 2 [Theorem Retrieval]: State the formal mathematical theorems required to solve the problem based on the visual features (e.g., Alternate Interior Angles Theorem).
Step 3 [Calculation]: Perform the step-by-step arithmetic needed.
Step 4 [Final Conclusion]: Output a short string with the final numerical or logical answer (e.g., \boxed{40}).

Do not deviate from 4 steps. Each step must be clearly numbered. Your response MUST contain all 4 steps."""

import base64
from io import BytesIO
from PIL import Image

async def process_problem(image_path: str, question: str, sem: asyncio.Semaphore) -> Dict:
    """Queries Gemini 3.1 Pro for a K=4 extraction, throttled by a semaphore."""
    async with sem:
        try:
            # Gemini API via AsyncOpenAI expects base64 encoded images inline if we don't have public URLs
            with open(image_path, "rb") as int_img:
                base64_img = base64.b64encode(int_img.read()).decode('utf-8')
                
            # Optional: Detect mime type (assuming jpeg or png from path)
            mime_type = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
            image_url = f"data:{mime_type};base64,{base64_img}"
            
            response = await client.chat.completions.create(
                model="gemini-3.1-pro-preview",
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            },
                        ],
                    }
                ],
                max_completion_tokens=2000,
            )
            print(f"✅ Generated reasoning for: {image_path}")
            return {
                "image_path": image_path,
                "question": question,
                "reasoning": response.choices[0].message.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

import argparse

async def main(dataset_path: str, output_path: str, images_dir: str, limit: int = None, target_sample: str = None):
    """Asynchronously processes a local JSONL dataset from GeoThought/playground using semaphore batching."""
    
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
    
    print(f"Loaded {len(problems)} total inference problems. Beginning batched asynchronous execution...")
    
    # Restrict concurrent API calls to batches of 5 at a time to stay under Gemini rate limits
    sem = asyncio.Semaphore(5)
    
    tasks = [process_problem(p["image_path"], p["question"], sem) for p in problems]
    valid_results = []
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_all_tokens = 0
    
    for i, future in enumerate(asyncio.as_completed(tasks)):
        r = await future
        if r is not None:
            valid_results.append(r)
            total_prompt_tokens += r.get("prompt_tokens", 0)
            total_completion_tokens += r.get("completion_tokens", 0)
            total_all_tokens += r.get("total_tokens", 0)
            
            # Print running totals and sample every 10 samples (or 50 for larger runs)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"\n--- Progress: {i+1}/{len(tasks)} Problems ---")
                print(f"Running Tokens -> Input: {total_prompt_tokens} | Output: {total_completion_tokens}")
                print(f"Sample Question: {r['question']}")
                print(f"Sample Reasoning:\n{r['reasoning']}")
                print("-" * 50)
    
    with open(output_path, 'w') as f:
        for res in valid_results:
            # We can strip the token usage from the jsonl if we only wanted it for counting, 
            # but it is harmless to leave it in.
            f.write(json.dumps(res) + '\n')
            
    print(f"\nSaved {len(valid_results)} extracted reasoning chains to {output_path}")
    print(f"--- FINAL TOKEN ACCOUNTING (Gemini 3.1 Pro Preview) ---")
    print(f"Input Tokens:  {total_prompt_tokens}")
    print(f"Output Tokens: {total_completion_tokens}")
    print(f"Total Tokens:  {total_all_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K=4 Euclidean Latent steps using Gemini")
    parser.add_argument("--dataset_path", type=str, default="GeoThought/playground/data/test_question.jsonl")
    parser.add_argument("--output_path", type=str, default="data/geothoughts_k4.jsonl")
    parser.add_argument("--images_dir", type=str, default="GeoThought/playground/data/")
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples to call API on")
    parser.add_argument("--target_sample", type=str, default=None, help="Specific sample to target (e.g., 'sample_13.jpg')")
    
    args = parser.parse_args()
    
    asyncio.run(main(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        images_dir=args.images_dir,
        limit=args.limit,
        target_sample=args.target_sample
    ))
