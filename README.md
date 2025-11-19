# UMBRA

- Paper (pre-print) : https://www.arxiv.org/abs/2511.11311
- Checkpoints: https://huggingface.co/pkoutsouvelis/mri-foundation_fomo60k/tree/main

## Data Directory Structure

The data should be organized in a hierarchical structure:

```
data_dir/
├── sub_1/
│   └── ses_1/
│       ├── t1.npy
│       ├── t1.yaml
│       ├── t1_2.npy
│       ├── t1_2.yaml
│       ├── flair.npy
│       └── flair.yaml
├── sub_2/
│   ├── ses_1/
│   │   ├── t1.npy
│   │   ├── dwi.npy
│   │   └── ...
│   └── ses_2/
│       └── ...
└── ...
```

- Each patient has a `sub_X` directory
- Each session has a `ses_Y` directory under the patient
- Modality files are named `{modality}.npy` or `{modality}_{number}.npy`
- Metadata can be in `.yaml` or `.pkl` files (optional)

## Data Modules

This project provides three PyTorch Lightning data modules for different training scenarios:

### 1. MAEDataModule (`data/mae_datamodule.py`)

**Purpose**: Masked Autoencoder (MAE) training only

**Dataset**: Uses `MAEDataset`
- Includes **ALL** scan types, including 'scan_*' files
- Returns individual volumes (not pairs)
- Each sample contains a single volume in the format: `{"volume": NDArray}`

**Usage**:
```python
from data.mae_datamodule import MAEDataModule

data_module = MAEDataModule(
    data_dir="/path/to/data",
    transforms=transforms,
    batch_size=10
)
```

**Returns**: Single dataloader for training and validation

---

### 2. ContrastiveDataModule (`data/contrastive_datamodule.py`)

**Purpose**: Contrastive learning training only

**Dataset**: Uses `ContrastivePatientDataset`
- **Excludes** 'scan_*' files
- Creates pairs of volumes from the same patient and session
- Each sample contains two volumes: `{"vol1": NDArray, "vol2": NDArray}`
- Applies shared random cropping to ensure spatial alignment

**Usage**:
```python
from data.contrastive_datamodule import ContrastiveDataModule

data_module = ContrastiveDataModule(
    data_dir="/path/to/data",
    transforms=transforms,
    batch_size=10
)
```

**Returns**: Single dataloader for training and validation

---

### 3. CombinedDataModule (`data/combined_datamodule.py`)

**Purpose**: Combined MAE + Contrastive learning training **without data overlap**

**Datasets**: Uses both `MAEDataset` and `ContrastivePatientDataset` with smart filtering
- Provides two separate dataloaders in a list with **NO overlapping images**
- First loader: Contrastive pairs (excludes 'scan_*' files)
- Second loader: MAE individual volumes that are NOT in contrastive pairs
  - Includes 'scan_*' files (always excluded from contrastive)
  - Includes single modality files (can't form pairs in contrastive)
  - **Excludes** multi-modality files (those are used in contrastive pairs)
- Supports different batch sizes for each dataset

**Usage**:
```python
from data.combined_datamodule import CombinedDataModule

data_module = CombinedDataModule(
    data_dir="/path/to/data",
    transforms=transforms,
    batch_size=10,          # for contrastive
    mae_batch_size=8        # for MAE (optional, defaults to batch_size)
)

# Returns list of two dataloaders
train_loaders = data_module.train_dataloader()
contrastive_loader, mae_loader = train_loaders
```

**Returns**: List of two dataloaders `[contrastive_loader, mae_loader]` for training and validation

**Important**: This module ensures zero overlap between datasets - no image appears in both the contrastive pairs and the MAE dataset, maximizing data efficiency.

---

## Common Features

All three data modules:
- Use 80/20 train/validation split with `random_state=42`
- Filter patients based on the same patient ID extraction logic
- Support optional transforms via the `transforms` parameter
- Use 32 workers for data loading
- Automatically handle train/test splits consistently across datasets

## Contrastive Learning Details

### Positive and Negative Pairs

Unlike the original MoCo paper (which assumes each sample is independent), this implementation handles **longitudinal medical imaging data** where the same patient can have multiple sessions.

**Positive pairs**: Different modalities from the **same patient AND same session**
- Example: (T1, FLAIR) from patient 5, session 1
- These are spatially aligned and share anatomical structure
- The model learns to bring these representations closer together

**Negative pairs**: Volumes from **different patients only**
- The model maintains a MoCo queue storing embeddings with patient/session metadata
- During contrastive loss computation, negatives from the same patient (any session) are **filtered out**
- Example: If processing patient 5 session 1, then patient 5 session 2 in the queue is **not used as a negative**
- Only embeddings from completely different patients (e.g., patient 7, patient 12) are used as negatives

### Why This Filtering Matters

**Without filtering** (original MoCo approach):
- ❌ Patient 5 session 1 would be pushed away from patient 5 session 2
- ❌ Model learns to distinguish sessions/timepoints of the same patient
- ❌ Representations become patient-specific rather than anatomy-focused

**With filtering** (this implementation):
- ✅ Patient 5 session 1 and session 2 are never used as negatives for each other
- ✅ Model learns patient-invariant, anatomy-focused representations
- ✅ Longitudinal scans from the same patient maintain similar representations
- ✅ Only true anatomical differences between patients drive the learning

This is essential for medical imaging where longitudinal follow-up is common and you want representations that generalize across time points.