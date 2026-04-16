import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

batch_id = "batch_69bb63d1ff10819098c0c70e76949586"
batch = client.batches.retrieve(batch_id)

print("status:", batch.status)
print("request_counts:", batch.request_counts)
print("output_file_id:", batch.output_file_id)
print("error_file_id:", batch.error_file_id)

if batch.output_file_id:
    content = client.files.content(batch.output_file_id)
    text = content.read().decode("utf-8")
    with open("recovered_outputs.jsonl", "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved successful outputs to recovered_outputs.jsonl")
else:
    print("No output_file_id present.")

if batch.error_file_id:
    content = client.files.content(batch.error_file_id)
    text = content.read().decode("utf-8")
    with open("recovered_errors.jsonl", "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved request errors to recovered_errors.jsonl")
else:
    print("No error_file_id present.")