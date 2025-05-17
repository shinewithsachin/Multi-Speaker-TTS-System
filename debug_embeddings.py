
import numpy as np
data = np.load('data/speaker_embeddings.npy', allow_pickle=True).item()
print(f"Loaded {len(data)} items.")

print("First 5 keys:")
for i, key in enumerate(data.keys()):
    print(f"- {key}")
    if i >= 4: break

example_key = list(data.keys())[0] 
print(f"\nDetails for key '{example_key}':")
print(f"- Value type: {type(data[example_key])}")
if isinstance(data[example_key], np.ndarray):
     print(f"- Value shape: {data[example_key].shape}")
     print(f"- Value dtype: {data[example_key].dtype}")