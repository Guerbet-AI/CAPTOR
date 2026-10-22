# CAPTOR

Paper: [CAPTOR: a vision-language foundation model for Chest-Abdomen-Pelvis Oncologic CT](https://arxiv.org/abs/2610.14226)

CAPTOR (Chest Abdomen Pelvis foundaTion model for Oncologic Radiology) is a CT medical vision-language model (VLM) for image feature extraction, suitable for a wide range of downstream tasks : disease detection, survival prediction (both necessitate training of classifiers), retrieval (no further training needed) etc.

Our pipeline uses a segmentation model to compute anatomical crop centers (chest, gastrointestinal tract, liver and biliary tree, pelvic organs etc). This process was used during training to enable the model to process whole Chest-Abdomen-Pelvis CT scans, and can be used in inference to make sure the needed region is seen by the model. 

Given one or more NIfTI CT studies and its associated reports, the pipeline:

1. Runs **TotalSegmentator** on each CT to locate anatomical regions (thorax, abdomen, pelvis, ...).
2. **Preprocesses** the volumes using those segmentations (region cropping / crop-center extraction).
3. **Trains** a contrastive image–text model (3D image encoder + clinical text encoder) or runs **inference** with a trained checkpoint.

```
images/ (NIfTI)
   │
   ▼
totalsegmentator_inference.py  ──► TS_outputs/ (segmentation masks)
   │
   ▼
preprocessing.py  ──► global_dict_crop_centers.pkl (per-region crop centers)
   │
   ├──► train.py  ──► checkpoints/ (trained model)
   │
   └──► inference.py  ──► inference_outputs/ (predictions / embeddings)
```

The architecture is based on the MERLIN architecture, with a 3D-inflated ResNet-152 (`i3res.py` / `inflate.py`, inflating standard 2D ImageNet weights into 3D convolutions) as image encoder and Clinical-Longformer as text encoder, trained together CLIP-style per anatomical region.


## Install

### If you want to use our full training/inference pipeline 

```bash
# Clone this repo
git clone https://github.com/Guerbet-AI/CAPTOR.git

# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the project root
uv sync

#To activate the env
source .venv/bin/activate #or simply prefix commands with `uv run`, e.g. `uv run scripts/training_pipeline.py ...`.
```

The full pipelines also need TotalSegmentator:

```bash
uv add TotalSegmentator
```

If you do not have internet access on the machine you want to run TotalSegmentator on:

1. Install TotalSegmentator (and set up the license) on a machine with internet.
2. Run TotalSegmentator for one subject on this machine. This downloads the weights and saves them to `~/.totalsegmentator`.
3. Copy the folder `~/.totalsegmentator` from this machine to the machine without internet.


### If you only want to use the model with your own code 

```bash
pip install git+https://github.com/Guerbet-AI/CAPTOR.git
```

## Training/inference pipelines 
### Expected data layout

- CT studies : **NIfTI files** in a single folder (`--images_dir`, default `./images`).
  
- Both pipelines need a pickle file to define which image to use for each region : Run and modify as needed the script `create_dict_std_to_region_to_series_reports.py` to create the needed pickle file (`--path_dict_std_to_region_to_series_reports`, default `./dict_std_to_region_to_series_reports.pkl`). The file has the following structure:

```python
{
    "<study_id>": {
        "series": {
            "<region>": "<series_name>",   # e.g. "THORAX": "series_003"
            ...
        },
        "report": {
            "<region>": "<report_text_for_that_region>",
            ...
        },
        "set": "TRAIN" | "VAL" | "TEST",
        "ct_types": {
            "<series_name>": "<ct_type>",  # for every series belonging to the study
            ...
        },
    },
    ...
}
```

Where `ct_type` is one of: `THORAX`, `ABDOMEN`, `PELVIS`, `TA`, `AP`, `TAP`, `UNKNOWN`.

- `series` maps each anatomical **region** of interest to the specific CT **series** (i.e. which NIfTI file / acquisition) that covers it.
- `report` maps each region to the corresponding portion of the radiology report describing it.
- `ct_types` records what body coverage each series in the study actually has (e.g. a `TAP` series covers thorax+abdomen+pelvis and can be reused for several regions).
- `set` assigns the study to a split, used during training.

The list of regions the model is trained/run on comes from `regions` in `src/captor/utils/utils_dict.py`, and can be restricted at training time via `--regions`.

### Training

```bash
uv run scripts/training_pipeline.py \
    --images_dir ./images \
    --segmentations_dir ./TS_outputs \
    --path_dict_std_to_region_to_series_reports ./dict_std_to_region_to_series_reports.pkl \
    --path_global_dict_crop_centers ./global_dict_crop_centers.pkl \
    --path_clinical_long_former None \ #change if you want to train from resnet152 and clinical longformer init weights. Weights are available on HF
    --path_weights_resnet152 None \ #change if you want to train from resnet152 and clinical longformer init weights. Weights are available on HF
    --weights_dir Guerbet-AI/CAPTOR \ #set to None if you want to train from resnet152 and clinical longformer init weights
    --model_type CAPTOR_30K \ #set to None if you want to train from resnet152 and clinical longformer init weights
    --dir_save_model ./checkpoints \
    --exp_name my_experiment \
```

### Inference

```bash
uv run scripts/inference_pipeline.py \
    --images_dir ./images \
    --segmentations_dir ./TS_outputs \
    --path_dict_std_to_region_to_series_reports ./dict_std_to_region_to_series_reports.pkl \
    --path_global_dict_crop_centers ./global_dict_crop_centers.pkl \
    --path_clinical_long_former None \
    --path_weights_resnet152 None \
    --weights_dir Guerbet-AI/CAPTOR \
    --model_type CAPTOR_30K \
    --dir_save_inference ./inference_outputs \
```

### Running individual stages

Each of `totalsegmentator_inference.py`, `preprocessing.py`, `train.py`, and `inference.py` can also be run directly (not just through the two pipeline orchestrators) — refer to each script's own `--help` / docstring for its specific arguments. This is useful, for example, to re-run only preprocessing after fixing the crop-center logic, without recomputing segmentations, or to run TotalSegmentator once on a shared dataset that both training and inference will reuse.

Both `training_pipeline.py` and `inference_pipeline.py` chain the same first two stages (TotalSegmentator inference, then preprocessing) before running their respective final stage. Each stage can be skipped with `--do_segmentations False` / `--do_preprocessing False` if its outputs already exist, which is useful for iterating on training/inference without recomputing segmentations every time.

The full list of arguments is at the end of the readme.


## Use without our preprocessing and pipelines

```python
from captor import MerlinArchitecture

model = MerlinArchitecture.from_pretrained(
    "Guerbet-AI/CAPTOR",          # HF repo id OR a local folder (see "Offline use")
    subfolder="CAPTOR_30K",       # which checkpoint to load
    device="cuda:0",              # default: "cpu"
    ImageEmbedding=True,         # default : False (embeddings from the shared image-text space (size 512)), for image feature extract set to True (size 2048)
    )
print(model)
```
The model is returned in **eval mode**. Call `model.train()` if you want to fine-tune it.

The input transformation needed is the following :
```python
Transform = monai.transforms.Compose(
    [
        LoadImaged(keys=["image"], image_only=False, meta_keys=["image_meta_dict"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),

    ]
)
```
To use our preprocessing pipeline and dataloader please see section II.

### Available checkpoints

Each checkpoint lives in its own subfolder of the Hugging Face repo [`Guerbet-AI/CAPTOR`](https://huggingface.co/Guerbet-AI/CAPTOR). Only the subfolder you request is downloaded.

| `subfolder` | Description |
|---|---|
| `CAPTOR_30K` | trained on ~30K CT studies from our proprietray dataset |
| `CAPTOR_100K` | trained on ~100K CT studies from our proprietray dataset and public datasets : Merlin, CT-RATE, INSPECT, BIMCV-R |


## Offline use (no internet on the target machine)

On a machine **with** internet, download only what you need:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "Guerbet-AI/CAPTOR",
    local_dir="./captor_weights",
    allow_patterns=["tokenizer/*", "CAPTOR_30K/*", "CAPTOR_100K/*"],
)
```

Copy the `captor_weights` folder to the offline machine, then:

```python
from captor import MerlinArchitecture

model = MerlinArchitecture.from_pretrained("./captor_weights", subfolder="CAPTOR_30K")
```

The folder must have this structure:

```
captor_weights/
├── tokenizer/               # shared by all checkpoints
└── CAPTOR_30K/
    ├── config.json
    └── model.safetensors
```

If the model is already in your Hugging Face cache, you can also set `HF_HUB_OFFLINE=1` (or pass `local_files_only=True`) and use the repo id as usual.

If you get `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`, your local path does not exist from the directory you run Python in. Check it with `os.path.isdir(path)`.




## Full list of arguments 
| Argument | Default | Description |
|---|---|---|
| `--images_dir` | `./images` | NIfTI images folder |
| `--segmentations_dir` | `./TS_outputs` | Where TotalSegmentator outputs are stored |
| `--do_segmentations` | `True` | Run TotalSegmentator if not already done |
| `--do_preprocessing` | `True` | Compute crop centers if not already done |
| `--workers` | `10` | Parallel workers for preprocessing |
| `--path_dict_std_to_region_to_series_reports` | `./dict_std_to_region_to_series_reports.pkl` | Study/series/report metadata (see §3) |
| `--path_global_dict_crop_centers` | `./global_dict_crop_centers.pkl` | Precomputed/output crop centers |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `18` | Batch size |
| `--learning_rate` | `1.25e-6` | Learning rate |
| `--patience` | `10` | Early-stopping patience (epochs without improvement) |
| `--train_logit_scale` | `False` | Whether the CLIP-style logit scale is a trainable parameter |
| `--path_clinical_long_former` | `./clinical_longformer` | Text encoder init weights |
| `--path_weights_resnet152` | `./weight_resnet_152.ckpt` | Image encoder init weights |
| `--weights_dir` | `Guerbet-AI/CAPTOR` | Optional full-model weights directory if you have downloaded them, for example './captor_weights/ |
| `--model_type` | `CAPTOR_30K` | If you want to start from pretrained weights, also accepts 'CAPTOR_100K', if set to anything else the text encoder and image encoder init weights paths will be used |
| `--dir_save_model` | `./` | Where to save trained checkpoints |
| `--exp_name` | `"test"` | Experiment name (used for checkpoint/log naming) |
| `--cache_dir` | `./cache` | Cache directory for dataset caching |
| `--regions` | `None` (all) | Restrict training to a subset of regions |





## Citations 
```bibtex
@misc{jeanmactoux2026captorvisionlanguagefoundationmodel,
      title={CAPTOR: a vision-language foundation model for Chest-Abdomen-Pelvis Oncologic CT}, 
      author={Séléna Jean-Mactoux and Sophie Loizillon and Ghina Jardali and Amandine Crombé and Nathalie Lassau and Alexandre Bône},
      year={2026},
      eprint={2610.14226},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.14226}, 
}
```
