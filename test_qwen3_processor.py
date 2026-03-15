import torch
from transformers import AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
image = Image.new('RGB', (224, 224), color = (73, 109, 137))
text = "Find the measure of angle X <thought_1><thought_2><thought_3><thought_4>"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image, # Qwen3-VL accepts PIL Images directly in the pipeline
            },
            {"type": "text", "text": text},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

print(inputs.keys())
print("input_ids shape:", inputs["input_ids"].shape)
if "pixel_values" in inputs:
    print("pixel_values shape:", inputs["pixel_values"].shape)
