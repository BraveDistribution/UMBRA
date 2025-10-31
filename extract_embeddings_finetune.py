"""
Extract embeddings from pretrained checkpoint for finetuning dataset (e.g., ISLES).

IMPORTANT: This script is designed to extract embeddings from a PRETRAINED checkpoint
(NOT a finetuned checkpoint) on a downstream finetuning dataset. This is ONLY safe
if the finetuning dataset was NOT included in the pretraining data.

For ISLES dataset, this extracts embeddings for 3 modalities: DWI, FLAIR, ADC.

Outputs:
  - embeddings.pt: Tensor of shape (N, 768) with all embeddings
  - metadata.csv: DataFrame with columns [filename, embedding_index, modality, patient_id, session_id]

Usage:
    # Extract from full dataset (safe if using pretrained checkpoint)
    python extract_embeddings_finetune.py --checkpoint /path/to/pretrain.ckpt \
                                           --data_dir /path/to/isles \
                                           --output_dir /path/to/output \
                                           --modalities dwi flair adc \
                                           --seed 42

    # Extract from specific split (e.g., test set only)
    python extract_embeddings_finetune.py --checkpoint /path/to/pretrain.ckpt \
                                           --data_dir /path/to/isles \
                                           --test_dir /path/to/isles_test \
                                           --output_dir /path/to/output \
                                           --modalities dwi flair adc \
                                           --split test \
                                           --seed 42
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Literal
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.finetuning_datamodule import FinetuningDataModule
from transforms.composed import get_segmentation_transforms
from models.foundation import ContrastiveMAEPretrainer
from models.finetuning import FinetuningModule


def extract_embeddings(
    checkpoint_path: Path,
    data_dir: Path,
    modalities: List[str],
    test_dir: Optional[Path] = None,
    split: Literal["full", "train", "val", "test"] = "full",
    input_size: int = 96,
    seed: int = 42,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Extract 768-dim embeddings from finetuning dataset.

    Extracts embeddings for EACH modality separately (model expects 1 channel input).

    Args:
        checkpoint_path: Path to PRETRAINED model checkpoint (NOT finetuned)
        data_dir: Path to finetuning data directory (hierarchical structure)
        modalities: List of modalities to extract (e.g., ['dwi', 'flair', 'adc'])
        test_dir: Optional path to separate test directory
        split: Which split to extract from ('full', 'train', 'val', 'test')
        input_size: Input size for model (default: 96)
        seed: Random seed for data splits (default: 42)
        device: Device to run on (default: 'cuda')

    Returns:
        embeddings: Tensor of shape (N_total, 768) where N_total = N_samples * N_modalities
        metadata: DataFrame with columns [filename, embedding_index, modality, patient_id, session_id]
    """
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device_obj}")

    # Load model
    print(f"\n{'='*80}")
    print(f"STEP 1/4: LOADING MODEL")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"⚠️  WARNING: Ensure this is a PRETRAINED checkpoint, NOT finetuned on this dataset!")
    try:
        model = ContrastiveMAEPretrainer.load_from_checkpoint(str(checkpoint_path))
        print("✓ Loaded as ContrastiveMAEPretrainer")
    except Exception as e1:
        try:
            model = FinetuningModule.load_from_checkpoint(str(checkpoint_path))
            print("✓ Loaded as FinetuningModule")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load checkpoint:\n"
                f"  As ContrastiveMAEPretrainer: {e1}\n"
                f"  As FinetuningModule: {e2}"
            )

    model = model.to(device_obj)
    model.eval()
    print(f"✓ Model moved to {device_obj} and set to eval mode")

    # Will collect embeddings from all modalities
    all_embeddings_global = []
    all_metadata_global = []

    print(f"\n{'='*80}")
    print(f"STEP 2/4: PROCESSING MODALITIES")
    print(f"{'='*80}")
    print(f"Data directory: {data_dir}")
    print(f"Test directory: {test_dir if test_dir else 'None (using train/test split)'}")
    print(f"Modalities to process: {modalities}")
    print(f"Split: {split}")
    print(f"Seed: {seed} (for data splits)")
    print(f"")
    print(f"NOTE: Processing each modality separately (model expects 1 channel)")
    print(f"{'='*80}\n")

    # Process each modality separately
    for modality_idx, modality_name in enumerate(modalities):
        print(f"\n{'─'*80}")
        print(f"Processing modality {modality_idx + 1}/{len(modalities)}: {modality_name.upper()}")
        print(f"{'─'*80}\n")

        # Create validation transforms for SINGLE modality
        val_transforms = get_segmentation_transforms(
            input_size=input_size,
            keys=[modality_name],  # Single modality only!
            seg_key="mask",
            out_key="volume",
            n_patches=1,
            n_pos=0,
            n_neg=0,
            val_mode=True,
        )

        # Configure splits based on extraction mode
        if split == "full":
            train_val_split = 0.0
            train_test_split = 0.0
            actual_test_dir = None
        elif split == "test":
            train_val_split = 0.0
            train_test_split = 0.2 if not test_dir else 0.0
            actual_test_dir = test_dir
        elif split == "val":
            train_val_split = 0.2
            train_test_split = 0.2 if not test_dir else 0.0
            actual_test_dir = test_dir
        elif split == "train":
            train_val_split = 0.2
            train_test_split = 0.2 if not test_dir else 0.0
            actual_test_dir = test_dir
        else:
            raise ValueError(f"Invalid split: {split}")

        data_module = FinetuningDataModule(
            data_dir=str(data_dir),
            train_transforms=val_transforms,
            val_transforms=val_transforms,
            modalities=[modality_name],  # Single modality!
            scan_type="numpy",
            target="mask",
            require_all_labels=False,
            require_all_scans=True,
            test_dir=str(actual_test_dir) if actual_test_dir else None,
            train_val_split=train_val_split,
            train_test_split=train_test_split,
            batch_size=1,
            num_workers=0,
            seed=seed,
        )

        # Setup and get appropriate dataloader
        if split == "full":
            data_module.setup('predict')
            loader = data_module.predict_dataloader()
            dataset_name = "full dataset"
        elif split == "test":
            data_module.setup('test')
            loader = data_module.test_dataloader()
            dataset_name = "test set"
        elif split == "val":
            data_module.setup('fit')
            loader = data_module.val_dataloader()
            dataset_name = "validation set"
        elif split == "train":
            data_module.setup('fit')
            loader = data_module.train_dataloader()
            dataset_name = "training set"

        print(f"✓ Dataset size: {len(loader.dataset)} samples")
        print(f"✓ Extracting from: {dataset_name}")

        # Extract embeddings for this modality
        total_samples = len(loader.dataset)
        print(f"\nExtracting embeddings for {modality_name.upper()}...")
        print(f"  Total samples: {total_samples}")
        print(f"  Device: {device_obj}")
        print(f"  Embedding dimension: 768\n")

        all_embeddings = []
        all_filenames = []
        all_patient_ids = []
        all_session_ids = []

        dataset = loader.dataset

        with torch.no_grad():
            for batch_idx, batch_list in enumerate(tqdm(loader, desc=f'{modality_name.upper()}', unit='sample')):
                # MONAI's list_data_collate returns a list of batches
                batch = batch_list[0] if isinstance(batch_list, list) else batch_list

                # Get volume from batch
                volume = batch['volume'].to(device_obj)

                # Add channel dimension if needed (B, D, H, W) -> (B, 1, D, H, W)
                if volume.ndim == 4:
                    volume = volume.unsqueeze(1)

                # Get metadata from batch
                patient_id = batch.get('patient', f'unknown_{batch_idx}')
                session_id = batch.get('session', 'unknown')

                # Convert to string
                if torch.is_tensor(patient_id):
                    patient_id = str(patient_id.item())
                elif isinstance(patient_id, list):
                    patient_id = str(patient_id[0]) if len(patient_id) > 0 else f'unknown_{batch_idx}'
                else:
                    patient_id = str(patient_id)

                if torch.is_tensor(session_id):
                    session_id = str(session_id.item())
                elif isinstance(session_id, list):
                    session_id = str(session_id[0]) if len(session_id) > 0 else 'unknown'
                else:
                    session_id = str(session_id)

                # Extract filename from dataset
                if hasattr(dataset, 'data_entries') and batch_idx < len(dataset.data_entries):
                    entry = dataset.data_entries[batch_idx]
                    if modality_name in entry:
                        filename = Path(entry[modality_name]).name
                    else:
                        filename = f'sample_{batch_idx}_{modality_name}.npy'
                else:
                    filename = f'sample_{batch_idx}_{modality_name}.npy'

                # Extract features from encoder
                if hasattr(model, 'encoder'):
                    features = model.encoder(volume)[-1]  # (B, 768, D, H, W)
                elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
                    features = model.model.encoder(volume)[-1]
                else:
                    raise AttributeError("Model does not have expected encoder structure")

                # Global average pooling
                features_pooled = F.adaptive_avg_pool3d(features, 1)  # (B, 768, 1, 1, 1)
                features_pooled = features_pooled.flatten(1)  # (B, 768)

                # Store
                all_embeddings.append(features_pooled.cpu())
                all_filenames.append(filename)
                all_patient_ids.append(patient_id)
                all_session_ids.append(session_id)

        # Concatenate embeddings for this modality
        if len(all_embeddings) == 0:
            print(f"⚠️  WARNING: No samples found for modality {modality_name}")
            continue

        embeddings_modality = torch.cat(all_embeddings, dim=0)  # (N, 768)

        # Create metadata for this modality
        metadata_modality = pd.DataFrame({
            'filename': all_filenames,
            'modality': [modality_name] * len(all_filenames),
            'patient_id': all_patient_ids,
            'session_id': all_session_ids,
        })

        print(f"✓ Extracted {len(embeddings_modality)} embeddings for {modality_name.upper()}")
        print(f"  Unique patients: {metadata_modality['patient_id'].nunique()}\n")

        # Add to global collections
        all_embeddings_global.append(embeddings_modality)
        all_metadata_global.append(metadata_modality)

    # Combine all modalities
    print(f"\n{'='*80}")
    print(f"STEP 3/4: COMBINING ALL MODALITIES")
    print(f"{'='*80}")

    if len(all_embeddings_global) == 0:
        raise RuntimeError("No embeddings extracted! Check your data directory and modalities.")

    embeddings = torch.cat(all_embeddings_global, dim=0)  # (N_total, 768)
    metadata = pd.concat(all_metadata_global, ignore_index=True)

    # Add embedding index
    metadata.insert(1, 'embedding_index', range(len(metadata)))

    print(f"\nCombined results:")
    print(f"  Total embeddings: {embeddings.shape[0]}")
    print(f"  Embeddings per modality:")
    for mod in modalities:
        count = (metadata['modality'] == mod).sum()
        print(f"    - {mod.upper()}: {count}")
    print(f"  Unique patients: {metadata['patient_id'].nunique()}")
    print(f"  Unique sessions: {metadata['session_id'].nunique()}")

    return embeddings, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from pretrained checkpoint for finetuning dataset"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PRETRAINED checkpoint file (.ckpt) - NOT finetuned'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to finetuning data directory (hierarchical: sub_X/ses_Y/*.npy)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for embeddings and metadata'
    )
    parser.add_argument(
        '--modalities',
        type=str,
        nargs='+',
        required=True,
        help='List of modalities to extract (e.g., dwi flair adc)'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default=None,
        help='Optional path to separate test directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='full',
        choices=['full', 'train', 'val', 'test'],
        help='Which split to extract from (default: full). Use "full" only with pretrained checkpoint!'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=96,
        help='Input size for model (default: 96)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for data splits (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (default: cuda)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    test_dir = Path(args.test_dir) if args.test_dir else None

    # Validate inputs
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if test_dir and not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Warning for full dataset extraction
    if args.split == "full":
        print("\n" + "="*80)
        print("⚠️  WARNING: FULL DATASET EXTRACTION MODE")
        print("="*80)
        print("You are extracting embeddings from the FULL dataset.")
        print("This is ONLY safe if:")
        print("  1. The checkpoint is from PRETRAINING (not finetuned on this data)")
        print("  2. This finetuning dataset was NOT included in pretraining")
        print("\nIf either condition is false, you WILL have data leakage!")
        print("="*80)
        response = input("\nDo you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*80)
    print("EMBEDDING EXTRACTION - FINETUNING DATASET")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data directory: {data_dir}")
    print(f"Test directory: {test_dir if test_dir else 'None'}")
    print(f"Output directory: {output_dir}")
    print(f"Modalities: {args.modalities}")
    print(f"Split: {args.split}")
    print(f"Input size: {args.input_size}")
    print(f"Seed: {args.seed}")
    print("="*80)

    # Extract embeddings
    embeddings, metadata = extract_embeddings(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        modalities=args.modalities,
        test_dir=test_dir,
        split=args.split,
        input_size=args.input_size,
        seed=args.seed,
        device=args.device,
    )

    # Save outputs
    embeddings_path = output_dir / 'embeddings.pt'
    metadata_path = output_dir / 'metadata.csv'

    print(f"\n{'='*80}")
    print(f"STEP 4/4: SAVING RESULTS")
    print(f"{'='*80}")
    print(f"Saving embeddings tensor...")
    torch.save(embeddings, embeddings_path)
    print(f"✓ Saved embeddings: {embeddings_path}")
    print(f"  Shape: {embeddings.shape}")

    print(f"\nSaving metadata CSV...")
    metadata.to_csv(metadata_path, index=False)
    print(f"✓ Saved metadata: {metadata_path}")
    print(f"  Rows: {len(metadata)}")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nTo load embeddings:")
    print(f"  import torch")
    print(f"  embeddings = torch.load('{embeddings_path}')")
    print(f"\nTo load metadata:")
    print(f"  import pandas as pd")
    print(f"  metadata = pd.read_csv('{metadata_path}')")
    print("="*80)


if __name__ == '__main__':
    main()
