import json
import os

results_file = "GeoThought/results/InternVL3-8B-Instruct_results.jsonl"
stats_file = "GeoThought/results/InternVL3-8B-Instruct_results_stats.json"

if not os.path.exists(results_file):
    print("Results file not found.")
    exit(1)

total = 0
correct = 0

with open(results_file, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            total += 1
            if data.get('is_correct'):
                correct += 1
        except:
            pass

stats = {
    "Total Samples": total,
    "Correct Samples": correct,
    "Accuracy (%)": round((correct / total) * 100, 2) if total > 0 else 0.0
}

with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=4)

print(json.dumps(stats, indent=4))
