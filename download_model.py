from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="speechbrain/spkrec-ecapa-voxceleb",
    local_dir="pretrained_models/spkrec",
    local_dir_use_symlinks=False  
)

print("✅ Model downloaded successfully to 'pretrained_models/spkrec'")
