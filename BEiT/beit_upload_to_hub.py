from huggingface_hub import HfApi

# python -m beit_upload_to_hub

model_dir = "final_model_finetuned"

api = HfApi()
api.create_repo(repo_id="parkouralvoil/cs180-BEiT-potato-disease", exist_ok=True)
api.upload_folder(
    folder_path=model_dir,
    commit_message="Add BEiT for potato disease images",
    repo_id="parkouralvoil/cs180-BEiT-potato-disease",
    repo_type="model",
)
print("ok")