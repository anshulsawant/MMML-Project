import torch
import os

src_dir = "target_tensors/target_tensors_v2_huber_mean_pooled"
dst_dir = "target_tensors/target_tensors_v13_2step"
os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if not fname.endswith(".pt"):
        continue
    tensor = torch.load(os.path.join(src_dir, fname), map_location="cpu", weights_only=True)
    sliced = tensor[[0, 3], :]  # thought_1 and thought_4
    torch.save(sliced, os.path.join(dst_dir, fname))
    
print(f"Done. Saved {len(os.listdir(dst_dir))} sliced tensors to {dst_dir}")