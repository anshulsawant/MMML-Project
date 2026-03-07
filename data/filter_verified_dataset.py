import json
import os

with open('data/eval_mismatches.json', 'r') as f:
    mismatches = json.load(f)

mismatched_images = set(item['image'] for item in mismatches)

input_file = 'data/geothoughts_k4_gemini3.1.jsonl'
output_file = 'data/geothoughts_verified.jsonl'

verified_count = 0
dropped_count = 0

print(f"Filtering {input_file}...")
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if not line.strip(): continue
        data = json.loads(line)
        if data['image_path'] in mismatched_images:
            dropped_count += 1
            continue
            
        outfile.write(line)
        verified_count += 1

print(f"Successfully wrote {verified_count} pristine traces to {output_file}")
print(f"Dropped {dropped_count} mismatched traces.")
