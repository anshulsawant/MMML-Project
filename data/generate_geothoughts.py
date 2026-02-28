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

YOU MUST strictly follow this K=4 format exactly:
Step 1 [Visual Parsing]: Extract all explicit and implicit geometric relationships visible in the image (e.g., parallel lines, specific angle measures, lengths).
Step 2 [Theorem Retrieval]: State the formal mathematical theorems required to solve the problem based on the visual features (e.g., Alternate Interior Angles Theorem).
Step 3 [Calculation]: Perform the step-by-step arithmetic needed.
Step 4 [Final Conclusion]: Output a short string with the final numerical or logical answer (e.g., \boxed{40}).

Do not deviate from 4 steps. Each step must be clearly numbered."""

import base64
from io import BytesIO
from PIL import Image

async def process_problem(image_path: str, question: str) -> Dict:
    """Queries Gemini 3.1 Pro for a K=4 extraction."""
    try:
        # Gemini API via AsyncOpenAI expects base64 encoded images inline if we don't have public URLs
        with open(image_path, "rb") as int_img:
            base64_img = base64.b64encode(int_img.read()).decode('utf-8')
            
        # Optional: Detect mime type (assuming jpeg or png from path)
        mime_type = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
        image_url = f"data:{mime_type};base64,{base64_img}"
        
        response = await client.chat.completions.create(
            model="gemini-3.1-pro",
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
            max_tokens=1000,
        )
        return {
            "image_path": image_path,
            "question": question,
            "reasoning": response.choices[0].message.content
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

async def main(dataset_path: str, output_path: str, images_dir: str):
    """Asynchronously processes a local JSONL dataset from GeoThought/playground."""
    
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
                
                if os.path.exists(full_image_path) and question_text:
                    problems.append({
                        "image_path": full_image_path,
                        "question": question_text
                    })
                
                # Limit initial testing batch to 10
                if len(problems) >= 10:
                    break
    
    tasks = [process_problem(p["image_path"], p["question"]) for p in problems]
    results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    
    with open(output_path, 'w') as f:
        for res in valid_results:
            f.write(json.dumps(res) + '\n')
            
    print(f"Saved {len(valid_results)} extracted reasoning chains to {output_path}")

if __name__ == "__main__":
    # Example usage for the local GeoThought baseline data structure:
    asyncio.run(main(
        dataset_path="GeoThought/playground/data/test_question.jsonl",
        output_path="geothoughts_k4.jsonl",
        images_dir="GeoThought/playground/data/"
    ))
