# CAPTOR

CAPTOR (Chest Abdomen Pelvis foundaTion model for Oncologic Radiology) is a CT medical vision-language model (VLM). Given one or more NIfTI CT studies and its associated reports, the pipeline:

1. Runs **TotalSegmentator** on each CT to locate anatomical regions (thorax, abdomen, pelvis, ...).
2. **Preprocesses** the volumes using those segmentations (region cropping / crop-center extraction).
3. **Trains** a contrastive image–text model (3D image encoder + clinical text encoder) or runs **inference** with a trained checkpoint.

The architecture is based on the MERLIN architecture, with a 3D-inflated ResNet-152 (`i3res.py` / `inflate.py`, inflating standard 2D ImageNet weights into 3D convolutions) as image encoder and Clinical-Longformer as text encoder, trained together CLIP-style per anatomical region.

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
│   └── test.py                       # tests
│
├── src/captor/                       # importable package (business logic)
│   ├── data/
│   │   ├── dataloaders.py            # PyTorch datasets / dataloaders
│   │   └── monai_transforms.py       # MONAI-based preprocessing/augmentation transforms
│   ├── models/
│   │   ├── build.py                  # model factory / assembly (image encoder + text encoder)
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

## 2. Environment setup

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

### External weights you need to provide

| Argument | Purpose |
|---|---|
| `--path_clinical_long_former` | Local path to (or HF hub id downloaded to) the Clinical-Longformer text encoder weights |
| `--path_weights_resnet152` | Checkpoint used to initialize the 3D-inflated ResNet-152 image encoder |
| `--path_initial_model_weights` | Optional: full CAPTOR checkpoint to warm-start training/inference from (instead of building from the two encoders above) |

For TotalSegmentator : 
pip install TotalSegmentator 

If you do not have internet access on the machine you want to run TotalSegmentator on:

Install TotalSegmentator [and set up the license] on a machine with internet.
Run TotalSegmentator for one subject on this machine. This will download the weights and save them to ~/.totalsegmentator.
Copy the folder ~/.totalsegmentator from this machine to the machine without internet.
TotalSegmentator should now work also on the machine without internet.
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
    --path_clinical_long_former ./clinical_longformer \
    --path_weights_resnet152 ./weight_resnet_152.ckpt \
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
| `--path_initial_model_weights` | `None` | Optional full-model checkpoint to resume/warm-start from |
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
    --path_clinical_long_former ./clinical_longformer \
    --path_weights_resnet152 ./weight_resnet_152.ckpt \
    --path_initial_model_weights ./checkpoints/my_experiment_best.ckpt \
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
| `--path_initial_model_weights` | `None` | Trained checkpoint to run inference with (typically required) |
| `--dir_save_inference` | `./` | Where inference outputs are saved |
| `--batch_size` | `18` | Batch size |
| `--cache_dir` | `./cache` | Cache directory for dataset caching |

### 4.3 Running individual stages

Each of `totalsegmentator_inference.py`, `preprocessing.py`, `train.py`, and `inference.py` can also be run directly (not just through the two pipeline orchestrators) — refer to each script's own `--help` / docstring for its specific arguments. This is useful, for example, to re-run only preprocessing after fixing the crop-center logic, without recomputing segmentations, or to run TotalSegmentator once on a shared dataset that both training and inference will reuse.

---

## 5. Notes

- `regions` (imported from `captor.utils.utils_dict`) defines the canonical list of anatomical regions the model is aware of; it drives both what gets cropped in preprocessing and what the model trains/infers on.
- The `--do_segmentations` / `--do_preprocessing` flags exist so that expensive upstream steps (TotalSegmentator, crop-center computation) aren't repeated unnecessarily across multiple training runs or inference calls on the same dataset.
- Warnings are silenced (`warnings.filterwarnings("ignore")`) in both pipeline scripts to keep console output focused on pipeline progress.

---

## 6. Quick reference: pipeline flow

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