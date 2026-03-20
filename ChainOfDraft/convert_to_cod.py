import asyncio
import json
import os
import io
import re
import base64
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from openai import AsyncOpenAI

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5-mini"
IMAGE_SAVE_DIR = "./geothought_images"
BATCH_REQUEST_FILE = "batch_requests.jsonl"
FINAL_QWEN_DATASET = "qwen3_vl_cod_dataset.jsonl"
ITEMS_PER_BATCH = 2000 # ~400 tokens/item * 2000 = ~800k tokens (safe for Tier 1 2M queue limit)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are an expert mathematician solving geometry problems. Think step by step, but only keep a minimum draft for each thinking step. Limit each step to 5 words or mathematical symbols at most. Do not write full sentences. Focus only on the essential formulas, values, or geometric transformations needed to progress. Return the final answer at the end of the response after a separator ####."""

def parse_original_solution(solution_text):
    """Extracts the verbose thought process and the final answer from the original solution."""
    think_match = re.search(r'<think>(.*?)</think>', solution_text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    # The remainder of the text is the final answer
    answer_text = re.sub(r'<think>.*?</think>', '', solution_text, flags=re.DOTALL).strip()
    return think_text, answer_text

def process_image(img, image_id):
    """Saves the image for Qwen3-VL and returns a compressed base64 string for OpenAI."""
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
    # Save original image locally for Qwen3-VL
    local_path = os.path.join(IMAGE_SAVE_DIR, f"{image_id}.jpg")
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(local_path, format="JPEG")
    
    # Create a compressed version for the OpenAI Batch API to save payload size & costs
    buffered = io.BytesIO()
    img_copy = img.copy()
    img_copy.thumbnail((512, 512)) # Downscale to fit within OpenAI's low-res/standard constraints
    img_copy.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return local_path, img_base64

async def process_chunk(chunk_indices, ds, chunk_id):
    """Processes a slice of the dataset to stay under token limits."""
    chunk_req_file = f"batch_requests_{chunk_id}.jsonl"
    print(f"\n--- Processing Chunk {chunk_id} ({len(chunk_indices)} items) ---")
    
    with open(chunk_req_file, "w", encoding="utf-8") as f:
        for idx in chunk_indices:
            row = ds[idx]
            problem = row.get("problem", "")
            solution = row.get("solution", "")
            img = row["image"] if "image" in row else row["images"]
            
            verbose_think, _ = parse_original_solution(solution)
            image_id = f"geothought_{idx:05d}"
            
            local_path, img_base64 = process_image(img, image_id)
            
            user_text = (
                f"Problem:\n{problem}\n\n"
                f"Verbose reasoning to compress:\n{verbose_think}\n\n"
                "Please compress the above reasoning into a Chain-of-Draft based on the system instructions. "
                "End your response with #### followed by the final answer."
            )
            
            request_body = {
                "custom_id": f"item_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                            ]
                        }
                    ],
                    "max_completion_tokens": 1000
                }
            }
            f.write(json.dumps(request_body) + "\n")
            
    print("Uploading chunk batch file to OpenAI...")
    with open(chunk_req_file, "rb") as f:
        batch_file = await client.files.create(file=f, purpose="batch")
    
    print("Creating Batch job...")
    batch_job = await client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print(f"Batch {batch_job.id} submitted! Polling...")
    while True:
        job = await client.batches.retrieve(batch_job.id)
        status = job.status
        print(f"Current Status: {status}")
        if status in ["completed", "failed", "cancelled", "expired"]:
            break
        await asyncio.sleep(60)
        
    if status != "completed":
        print(f"Chunk failed! Status: {status}")
        if job.errors and job.errors.data:
            print("\nError Details:")
            for error in job.errors.data:
                print(f" - [{error.code}] {error.message}")
        raise Exception(f"Batch {batch_job.id} failed due to rate limits or formatting.")
        
    print("Chunk completed! Downloading and formatting results...")

    completed_n = getattr(job.request_counts, "completed", 0) if job.request_counts else 0
    failed_n = getattr(job.request_counts, "failed", 0) if job.request_counts else 0
    total_n = getattr(job.request_counts, "total", 0) if job.request_counts else 0

    print(f"Request counts: completed={completed_n}, failed={failed_n}, total={total_n}")
    print(f"output_file_id={job.output_file_id}")
    print(f"error_file_id={job.error_file_id}")

    success_lines = []
    error_lines = []

    if job.output_file_id:
        file_response = await client.files.content(job.output_file_id)
        success_text = file_response.read().decode("utf-8").strip()
        if success_text:
            success_lines = success_text.split("\n")

    if job.error_file_id:
        err_response = await client.files.content(job.error_file_id)
        err_text = err_response.read().decode("utf-8").strip()
        if err_text:
            error_lines = err_text.split("\n")

    if not success_lines:
        print("No successful outputs in this batch.")
    if error_lines:
        print(f"Found {len(error_lines)} error records. Writing them to a log file.")
    
    with open(f"batch_errors_{chunk_id}.jsonl", "w", encoding="utf-8") as ef:
        for line in error_lines:
            ef.write(line + "\n")
    file_response = await client.files.content(job.output_file_id)
    results = file_response.read().decode('utf-8').strip().split('\n')
    
    with open(FINAL_QWEN_DATASET, "a", encoding="utf-8") as f:
        for line in success_lines:
            if not line.strip():
                continue

            data = json.loads(line)
            idx = int(data["custom_id"].split("_")[1])
            image_id = f"geothought_{idx:05d}"
            local_path = os.path.join(IMAGE_SAVE_DIR, f"{image_id}.jpg")
            problem_text = ds[idx].get("problem", "Solve the geometry problem.")

            if data["response"]["status_code"] != 200:
                continue

            response_content = data["response"]["body"]["choices"][0]["message"]["content"]
            parts = response_content.split("####")
            cod_draft = parts[0].strip()
            final_ans = parts[1].strip() if len(parts) > 1 else ""

            assistant_content = f"<think>\n{cod_draft}\n</think>\n{final_ans}"

            qwen_format = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": local_path},
                            {"type": "text", "text": problem_text}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ]
            }
            f.write(json.dumps(qwen_format, ensure_ascii=False) + "\n")
                
    os.remove(chunk_req_file)

async def main():
    print("Loading GeoThought dataset...")
    ds = load_dataset("xinlingdedeng/Geo-Thought", split="train")
    total_items = len(ds)
    
    # 1. Figure out which items we already successfully completed 
    processed_indices = set()
    if os.path.exists(FINAL_QWEN_DATASET):
        with open(FINAL_QWEN_DATASET, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                img_path = data["messages"][0]["content"][0]["image"]
                idx_match = re.search(r'geothought_(\d+)', img_path)
                if idx_match:
                    processed_indices.add(int(idx_match.group(1)))
                    
    print(f"Found {len(processed_indices)} items already completed.")
    
    # 2. Get the remaining items
    unprocessed_indices = [i for i in range(total_items) if i not in processed_indices]
    
    if not unprocessed_indices:
        print("All items processed!")
        return
        
    # 3. Process them in batches
    for i in range(0, len(unprocessed_indices), ITEMS_PER_BATCH):
        chunk_indices = unprocessed_indices[i:i + ITEMS_PER_BATCH]
        chunk_id = i // ITEMS_PER_BATCH + 1
        await process_chunk(chunk_indices, ds, chunk_id)
        
    print(f"\nAll chunks processed! Formatted Qwen3-VL dataset saved to {FINAL_QWEN_DATASET}")

if __name__ == "__main__":
    asyncio.run(main())
