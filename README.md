# CAPTOR

Paper: [CAPTOR: a vision-language foundation model for Chest-Abdomen-Pelvis Oncologic CT](https://arxiv.org/abs/2610.14226)

CAPTOR (Chest Abdomen Pelvis foundaTion model for Oncologic Radiology) is a CT medical vision-language model (VLM). Given one or more NIfTI CT studies and its associated reports, the pipeline:

1. Runs **TotalSegmentator** on each CT to locate anatomical regions (thorax, abdomen, pelvis, ...).
2. **Preprocesses** the volumes using those segmentations (region cropping / crop-center extraction).
3. **Trains** a contrastive image–text model (3D image encoder + clinical text encoder) or runs **inference** with a trained checkpoint.

The architecture is based on the MERLIN architecture, with a 3D-inflated ResNet-152 (`i3res.py` / `inflate.py`, inflating standard 2D ImageNet weights into 3D convolutions) as image encoder and Clinical-Longformer as text encoder, trained together CLIP-style per anatomical region.

---

## Quick start without cloning the repo

### 1. Install

```bash
pip install git+https://github.com/Guerbet-AI/CAPTOR.git
```

(Replace the URL with the actual GitHub repository URL. With `uv`: `uv add git+https://github.com/Guerbet-AI/CAPTOR.git`.)

### 2. Load a trained model

```python
from captor import MerlinArchitecture

model = MerlinArchitecture.from_pretrained("Guerbet-AI/CAPTOR", subfolder="CAPTOR_30K")
print(model)
```

That is all you need. The weights, the model configuration and the tokenizer are downloaded automatically from the Hugging Face Hub and cached. **You do not need to download the ResNet-152 or Clinical-Longformer initialization weights**: the released checkpoints already contain every trained parameter.

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
To use our preprocessing pipeline and dataloader please see section 4.

### Available checkpoints

Each checkpoint lives in its own subfolder of the Hugging Face repo [`Guerbet-AI/CAPTOR`](https://huggingface.co/Guerbet-AI/CAPTOR). Only the subfolder you request is downloaded.

| `subfolder` | Description |
|---|---|
| `CAPTOR_30K` | ~30K CT studies from our proprietray dataset |
| `CAPTOR_100K` | ~100K CT studies from our proprietray dataset and public datasets : Merlin, CT-RATE, INSPECT, BIMCV-R |

### `from_pretrained` options

```python
model = MerlinArchitecture.from_pretrained(
    "Guerbet-AI/CAPTOR",          # HF repo id OR a local folder (see "Offline use")
    subfolder="CAPTOR_30K",       # which checkpoint to load
    device="cuda:0",              # default: "cpu"
    ImageEmbedding=False,         # see "Image-only embeddings" below
    strict=True,                  # set False if your weights do not match exactly
)
```

Notes:

- The model is returned in **eval mode**. Call `model.train()` if you want to fine-tune it.
- Extra keyword arguments (e.g. `ImageEmbedding=True`) override the values stored in the checkpoint's `config.json`. Do not change architecture fields such as `nb_channels` unless you know the weights still match.

### Image-only embeddings

To extract image embeddings only (no text input):

```python
model = MerlinArchitecture.from_pretrained(
    "Guerbet-AI/CAPTOR", subfolder="CAPTOR_30K", ImageEmbedding=True
)
```

In this mode the model may not contain all the heads present in the full checkpoint. If loading complains about unexpected keys, pass `strict=False` and check that the printed `unexpected` list contains only head weights and that `missing` is empty.

### Offline use (no internet on the target machine)

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

### TotalSegmentator

The full pipelines below (not `from_pretrained`) also need TotalSegmentator:

```bash
pip install TotalSegmentator
```

If you do not have internet access on the machine you want to run TotalSegmentator on:

1. Install TotalSegmentator (and set up the license) on a machine with internet.
2. Run TotalSegmentator for one subject on this machine. This downloads the weights and saves them to `~/.totalsegmentator`.
3. Copy the folder `~/.totalsegmentator` from this machine to the machine without internet.

TotalSegmentator should now also work on the machine without internet.

---

## 1. Project layout

```
CAPTOR/
├── scripts/                          # CLI entry points and pipeline logic
│   ├── training_pipeline.py          # full pipeline: segmentation → preprocessing → training
│   ├── inference_pipeline.py         # full pipeline: segmentation → preprocessing → inference
│   ├── totalsegmentator_inference.py # runs TotalSegmentator over a directory of images
│   ├── preprocessing.py              # builds crop centers / region crops from segmentations
│   ├── train.py                      # training loop for the contrastive model
│   ├── inference.py                  # inference loop / embedding extraction
│   ├── export_to_hf.py               # converts a trained checkpoint to the HF format and uploads it
│   └── test.py                       # tests
│
├── src/captor/                       # importable package (business logic)
│   ├── data/
│   │   ├── dataloaders.py            # PyTorch datasets / dataloaders
│   │   └── monai_transforms.py       # MONAI-based preprocessing/augmentation transforms
│   ├── models/
│   │   ├── build.py                  # MerlinArchitecture: model assembly, from_pretrained / save_pretrained
│   │   ├── i3res.py                  # inflated 3D ResNet architecture
│   │   ├── inflate.py                # utilities to inflate 2D weights into 3D
│   │   └── load.py                   # checkpoint / pretrained weight loading
│   └── utils/
│       ├── utils_dict.py             # shared constants, incl. `regions`
│       ├── utils_inference.py        # inference helpers
│       ├── utils_preprocessing.py    # preprocessing helpers
│       └── utils_train.py            # training helpers (losses, schedulers, logging, ...)
│
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

Every script under `scripts/` is documented with its own docstrings/argparse help, and can be run standalone or as part of the two orchestrating pipelines described below.

---

## 2. Environment setup (development / training)

If you only want to load the model, `pip install git+https://...` (see Quick start) is enough. To run the full pipelines or train, clone the repository.

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management (see `pyproject.toml` / `uv.lock`).

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the project root
uv sync
```

This creates a `.venv` using the Python version pinned in `.python-version` and installs all dependencies locked in `uv.lock`.

Activate it with:

```bash
source .venv/bin/activate
```

or simply prefix commands with `uv run`, e.g. `uv run scripts/training_pipeline.py ...`.

### Which weights do you need?

| You want to... | Weights needed |
|---|---|
| Run inference / extract embeddings with a released checkpoint | **None to download manually.** `from_pretrained` fetches everything (or use a local folder, see "Offline use") |
| Fine-tune or use your own `.pt` on the CAPTOR architecture | Released config + tokenizer (via `from_pretrained(..., weights_path=...)`) |
| Train from scratch | Initialization weights below |
| Export a new checkpoint to the Hugging Face format | Initialization weights below, plus your trained checkpoint |

Initialization weights (only for training from scratch or exporting):

| Argument | Purpose |
|---|---|
| `--path_clinical_long_former` | Local path to the Clinical-Longformer text encoder weights |
| `--path_weights_resnet152` | Checkpoint used to initialize the 3D-inflated ResNet-152 image encoder |

In code, the two ways of building the model are:

```python
# Trained checkpoint (inference, embeddings, fine-tuning): no init weights needed
model = MerlinArchitecture.from_pretrained("Guerbet-AI/CAPTOR", subfolder="CAPTOR_30K")

# From scratch (training): needs the initialization weights
model = MerlinArchitecture(
    ImageEmbedding=False,
    path_clinical_long_former="./clinical_longformer",
    path_weights_resnet152="./weight_resnet_152.ckpt",
)
```

---

## 3. Expected data layout

### CT volumes

CT studies must be provided as **NIfTI files** in a single folder (`--images_dir`, default `./images`). `totalsegmentator_inference.py` and `preprocessing.py` iterate over this directory.

### TotalSegmentator outputs

Segmentation masks are written to (and, if already computed, read from) `--segmentations_dir` (default `./TS_outputs`). Set `--do_segmentations False` in either pipeline to skip this step if segmentations already exist there.

### Series / report metadata dictionary

Both pipelines need a pickle file (`--path_dict_std_to_region_to_series_reports`, default `./dict_std_to_region_to_series_reports.pkl`) with the following structure:

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

### Crop centers dictionary

`preprocessing.py` derives, from the TotalSegmentator masks, a dictionary of crop centers (one 3D bounding-box center per region) and saves it to `--path_global_dict_crop_centers` (default `./global_dict_crop_centers.pkl`). This is what `train.py` / `inference.py` actually use to crop each volume to the relevant region at load time. Set `--do_preprocessing False` to reuse an existing one.

---

## 4. Usage

Both `training_pipeline.py` and `inference_pipeline.py` chain the same first two stages (TotalSegmentator inference, then preprocessing) before running their respective final stage. Each stage can be skipped with `--do_segmentations False` / `--do_preprocessing False` if its outputs already exist, which is useful for iterating on training/inference without recomputing segmentations every time.

### 4.1 Training

```bash
uv run scripts/training_pipeline.py \
    --images_dir ./images \
    --segmentations_dir ./TS_outputs \
    --path_dict_std_to_region_to_series_reports ./dict_std_to_region_to_series_reports.pkl \
    --path_global_dict_crop_centers ./global_dict_crop_centers.pkl \
    --path_clinical_long_former None \
    --path_weights_resnet152 None \
    --weights_dir Guerbet-AI/CAPTOR \
    --model_type CAPTOR_30K \
    --dir_save_model ./checkpoints \
    --exp_name my_experiment \
    --epochs 200 \
    --batch_size 18 \
    --learning_rate 1.25e-6 \
    --patience 10
```

Key arguments (see `training_pipeline.py --help` for the full list):

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

### 4.2 Inference

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
    --batch_size 18
```

Key arguments (see `inference_pipeline.py --help` for the full list):

| Argument | Default | Description |
|---|---|---|
| `--images_dir` | `./images` | NIfTI images folder |
| `--segmentations_dir` | `./TS_outputs` | Where TotalSegmentator outputs are stored |
| `--do_segmentations` | `True` | Run TotalSegmentator if not already done |
| `--do_preprocessing` | `True` | Compute crop centers if not already done |
| `--workers` | `10` | Parallel workers for preprocessing |
| `--path_dict_std_to_region_to_series_reports` | `./dict_std_to_region_to_series_reports.pkl` | Study/series/report metadata (see §3) |
| `--path_global_dict_crop_centers` | `./global_dict_crop_centers.pkl` | Precomputed/output crop centers |
| `--path_clinical_long_former` | `./clinical_longformer` | Text encoder init weights |
| `--path_weights_resnet152` | `./weight_resnet_152.ckpt` | Image encoder init weights |
| `--weights_dir` | `Guerbet-AI/CAPTOR` | Optional full-model weights directory if you have downloaded them, for example './captor_weights/ |
| `--model_type` | `CAPTOR_30K` | If you want to start from pretrained weights, also accepts 'CAPTOR_100K', if set to anything else the text encoder and image encoder init weights paths will be used |
| `--dir_save_inference` | `./` | Where inference outputs are saved |
| `--batch_size` | `18` | Batch size |
| `--cache_dir` | `./cache` | Cache directory for dataset caching |

### 4.3 Running individual stages

Each of `totalsegmentator_inference.py`, `preprocessing.py`, `train.py`, and `inference.py` can also be run directly (not just through the two pipeline orchestrators) — refer to each script's own `--help` / docstring for its specific arguments. This is useful, for example, to re-run only preprocessing after fixing the crop-center logic, without recomputing segmentations, or to run TotalSegmentator once on a shared dataset that both training and inference will reuse.

---

## 6. Notes

- `regions` (imported from `captor.utils.utils_dict`) defines the canonical list of anatomical regions the model is aware of; it drives both what gets cropped in preprocessing and what the model trains/infers on.
- The `--do_segmentations` / `--do_preprocessing` flags exist so that expensive upstream steps (TotalSegmentator, crop-center computation) aren't repeated unnecessarily across multiple training runs or inference calls on the same dataset.
- Warnings are silenced (`warnings.filterwarnings("ignore")`) in both pipeline scripts to keep console output focused on pipeline progress.

---

## 7. Quick reference: pipeline flow

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
