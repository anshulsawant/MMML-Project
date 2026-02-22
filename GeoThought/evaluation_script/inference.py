import requests
import base64
import os
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import argparse
import pandas as pd
import concurrent.futures
from multiprocessing import Manager, Lock
import time
import re
import sys
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
# ========================
# 1. Helper Functions
# ========================
def safe_parse(text):
    """Safely parse the answer from model output"""
    try:
        # Try to match <answer> tag
        answer_match = re.search(r"<answer>\s*([\d.]+)", text, re.IGNORECASE)
        if answer_match:
            return [float(answer_match.group(1))]

        # Try to match numbers in text
        numbers = re.findall(r"\d+\.?\d*", text)
        return [float(numbers[-1])] if numbers else None
    except:
        return None

def safe_verify(pred, truth, tolerance=1e-3):
    """Verify if the predicted answer is correct"""
    if not pred or not truth:
        return 0.0
    return 1.0 if abs(pred[0] - truth[0]) < tolerance else 0.0

# ========================
# 2. Vision-Language Message Client
# ========================
class VLMessageClient:
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name
        self.session = requests.Session()

    def _encode_image(self, image_path):
        """Encode image to base64"""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def build_messages(self, item, image_root):
        """Build messages"""
        # 1. Get image path
        image_path = os.path.join(image_root, item['image_path'].lstrip('./'))

        # 2. Build messages (modified as required)
        return [
            {
                "role": "system",
                "content": (
                    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."
                )
            },
            {
                "role": "user",
                "content": f"<image>\n{item['question']}"
            }
        ]

    def process_item(self, item, image_root, output_file, error_file, total_counter, correct_counter, lock):
        """Process a single item"""
        max_retries = 5
        attempt = 0
        result = None

        while attempt < max_retries:
            try:
                attempt += 1
                messages = self.build_messages(item, image_root)

                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "top_p": 1,
                    "repetition_penalty": 1.00
                }

                response = self.session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=100 + attempt * 5
                )
                response.raise_for_status()

                output = response.json()["choices"][0]["message"]["content"]

                # Parse and verify answer
                gt = safe_parse(item["ground_truth"])
                pred = safe_parse(output)
                is_correct = bool(pred and gt and safe_verify(pred, gt))

                # Build success result
                result = {
                    "question": str(item["question"]),
                    "image_path": str(item["image_path"]),
                    "model_output": str(output),
                    "extracted_answer": str(pred[0]) if pred else None,
                    "ground_truth": str(item["ground_truth"]),
                    "is_correct": bool(is_correct),
                    "attempt": int(attempt),
                    "success": bool(True),
                    "model": self.model_name
                }

                # Write to success file
                with lock:
                    try:
                        with open(output_file, "a") as f:
                            json.dump(result, f, ensure_ascii=False, default=str)
                            f.write("\n")
                            f.flush()
                    except Exception as e:
                        print(f"Failed to write to success file: {str(e)}")

                return (True, is_correct)

            except Exception as e:
                if attempt == max_retries:
                    # Build failure result
                    error_data = {
                        "question": str(item["question"]),
                        "image_path": str(item["image_path"]),
                        "error": str(e),
                        "attempt": int(attempt),
                        "success": bool(False)
                    }
                    # Write to error file
                    with lock:
                        try:
                            with open(error_file, "a") as f:
                                json.dump(error_data, f, ensure_ascii=False, default=str)
                                f.write("\n")
                                f.flush()
                        except Exception as e:
                            print(f"Failed to write to error file: {str(e)}")
                    return (False, False)
                else:
                    time.sleep(min(2 ** attempt, 10))

        return (False, False)

# ========================
# 3. Main Function
# ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://127.0.0.1:8000")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--prompt_path", required=True, help="Test set path")
    parser.add_argument("--image_root", default="../", help="Image root directory")
    parser.add_argument("--output_path", required=True, help="Output file path")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of worker processes")
    args = parser.parse_args()

    # Set error file path
    error_output_path = os.path.splitext(args.output_path)[0] + "_errors.jsonl"

    # 1. Load test data
    test_data_all = pd.read_json(args.prompt_path, lines=True).to_dict("records")
    total_samples = len(test_data_all)
    print(f"Total Test Samples: {total_samples}")
    print(f"Using Model: {args.model_name}")

    # 2. Recover successfully processed data (only records where success is true)
    processed_success = set()  # Successfully processed data
    recovered_total = 0
    recovered_correct = 0
    valid_records = []  # Valid success records

    # Checking output file...在
    output_file_exists = os.path.exists(args.output_path)

    # Recover success records and clean output file
    if output_file_exists:
        print("Checking output file...")
        with open(args.output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("success", False):
                        processed_success.add(data["image_path"])
                        recovered_total += 1
                        if data.get("is_correct", False):
                            recovered_correct += 1
                        valid_records.append(data)
                except json.JSONDecodeError:
                    continue

        # Rewrite output file, keeping only success records
        if valid_records:
            print(f"Found {len(valid_records)} successful records")
            with open(args.output_path, "w") as f:
                for record in valid_records:
                    json.dump(record, f, ensure_ascii=False, default=str)
                    f.write("\n")
        else:
            print("No valid records found in output file, creating empty file")
            open(args.output_path, 'w').close()
    else:
        # Create output file
        print("Output file does not exist, creating new file")
        open(args.output_path, 'w').close()

    # Safely print recovery statistics
    print(f"Successfully recovered records: {recovered_total} ({recovered_total/total_samples:.2%})")
    if recovered_total > 0:
        print(f"Recovered correct records: {recovered_correct} ({recovered_correct/recovered_total:.2%})")
    else:
        print(f"Recovered correct records: 0 (N/A)")

    # 3. Determine remaining data to process (all unsuccessful data)
    remaining_data = []
    for item in test_data_all:
        img_path = item["image_path"]
        if img_path not in processed_success:
            remaining_data.append(item)

    print(f"Remaining records to process: {len(remaining_data)}")

    # 4. Process remaining data
    if remaining_data:
        print(f"Started processing remaining records, using {args.max_workers} workers...")

        # Ensure error file exists
        if not os.path.exists(error_output_path):
            open(error_output_path, 'w').close()

        with Manager() as manager:
            total_counter = manager.Value('i', recovered_total)
            correct_counter = manager.Value('i', recovered_correct)
            lock = manager.Lock()

            with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []
                client = VLMessageClient(args.api_url, args.model_name)

                for item in remaining_data:
                    futures.append(
                        executor.submit(
                            client.process_item,
                            item=item,
                            image_root=args.image_root,
                            output_file=args.output_path,
                            error_file=error_output_path,
                            total_counter=total_counter,
                            correct_counter=correct_counter,
                            lock=lock
                        )
                    )

                # 5. Progress Bar
                with tqdm(total=len(remaining_data), desc="Processing Progress") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            success, is_correct = future.result()
                            if success:
                                with lock:
                                    total_counter.value += 1
                                    correct_counter.value += int(is_correct)
                        except Exception as e:
                            print(f"Processing Error: {str(e)}")
                        finally:
                            pbar.update(1)
                            current_total = total_counter.value
                            current_correct = correct_counter.value
                            processed_info = f"{current_total}/{total_samples}"

                            # Avoid division by zero error
                            if current_total > 0:
                                accuracy_info = f"{current_correct/current_total:.2%}"
                            else:
                                accuracy_info = "N/A"

                            pbar.set_postfix({
                                "Correct": current_correct,
                                "Total": current_total,
                                "Accuracy": accuracy_info,
                                "Processed": processed_info
                            })
                            
                            # Output running stats for validation
                            print(f"\n[Running Validation] Processed: {current_total}/{total_samples} | Correct: {current_correct} | Accuracy: {accuracy_info}", flush=True)

    # 6. Final Statistics
    # Count success file
    success_count = 0
    correct_count = 0
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("success", False):
                        success_count += 1
                        if data.get("is_correct", False):
                            correct_count += 1
                except:
                    continue

    # Count error file
    error_count = 0
    if os.path.exists(error_output_path):
        with open(error_output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if not data.get("success", True):
                        error_count += 1
                except:
                    continue

    total_processed = success_count + error_count

    # Avoid division by zero error
    if success_count > 0:
        final_accuracy = correct_count / success_count
    else:
        final_accuracy = 0

    print("\nFinal Statistics:")
    print(f"Total Test Samples: {total_samples}")
    print(f"Successfully processed records: {success_count} ({success_count/total_samples:.2%})")
    print(f"Failed to process records: {error_count} ({error_count/total_samples:.2%})")

    if success_count > 0:
        print(f"Correct results: {correct_count} ({final_accuracy:.2%})")
    else:
        print(f"Correct results: 0 (N/A)")

    print(f"Total processed records: {total_processed} (Should match total test samples: {'Yes' if total_processed == total_samples else 'No'})")

    # Save statistics file
    stats_path = os.path.splitext(args.output_path)[0] + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_samples": total_samples,
            "model_name": args.model_name,
            "successful_inferences": success_count,
            "failed_inferences": error_count,
            "correct_results": correct_count,
            "accuracy": final_accuracy,
            "output_file": args.output_path,
            "error_file": error_output_path
        }, f, indent=4)

    print(f"Statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()
