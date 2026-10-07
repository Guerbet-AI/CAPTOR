import argparse, os, tempfile, torch
from huggingface_hub import HfApi
from captor import MerlinArchitecture

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--name", required=True)                 # subfolder name, e.g. "ct_chest"
p.add_argument("--path_clinical_long_former", required=True)
p.add_argument("--path_weights_resnet152", required=True)
p.add_argument("--repo_id", required=True)
p.add_argument("--upload_tokenizer", action="store_true")   # only needed once
args = p.parse_args()

model = MerlinArchitecture(ImageEmbedding=False,
                           path_clinical_long_former=args.path_clinical_long_former,
                           path_weights_resnet152=args.path_weights_resnet152)
ckpt = torch.load(args.checkpoint, map_location="cpu")
ckpt = ckpt.get("model_state_dict", ckpt)
ckpt = {k.removeprefix("module."): v for k, v in ckpt.items()}
print(model.load_state_dict(ckpt, strict=False))        # inspect missing/unexpected keys

api = HfApi()
api.create_repo(args.repo_id, exist_ok=True, private=True)

with tempfile.TemporaryDirectory() as tmp:
    model.save_pretrained(tmp, save_tokenizer=False)
    api.upload_folder(folder_path=tmp, repo_id=args.repo_id, path_in_repo=args.name)
    if args.upload_tokenizer:
        model.encode_text.tokenizer.save_pretrained(os.path.join(tmp, "tokenizer"))
        api.upload_folder(folder_path=os.path.join(tmp, "tokenizer"),
                          repo_id=args.repo_id, path_in_repo="tokenizer")