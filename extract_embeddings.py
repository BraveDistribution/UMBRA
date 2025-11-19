r"""
Extract embeddings from pretrained checkpoint for validation dataset.

Outputs:
  - embeddings.pt: Tensor of shape (N, 768) with all embeddings
  - metadata.csv: DataFrame with columns [filename, embedding_index, modality, patient_id, session_id]

Usage:
    python extract_embeddings.py --checkpoint /path/to/checkpoint.ckpt \
                                  --data_dir /path/to/pretrain_parsed \
                                  --output_dir /path/to/output \
                                  --seed 42
"""

import argparse
from pathlib import Path
from typing import Tuple, List
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.contrastive_datamodule import ContrastiveDataModule
from transforms.composed import get_contrastive_transforms
from models.foundation import ContrastiveMAEPretrainer
from models.finetuning import FinetuningModule


def extract_embeddings(
    checkpoint_path: Path,
    data_dir: Path,
    input_size: int = 96,
    seed: int = 42,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Extract 768-dim embeddings from validation dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to pretrain data directory (hierarchical structure)
        input_size: Input size for model (default: 96)
        seed: Random seed for validation split (default: 42)
        device: Device to run on (default: 'cuda')

    Returns:
        embeddings: Tensor of shape (N, 768)
        metadata: DataFrame with columns [filename, embedding_index, modality, patient_id, session_id]
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\n{'='*80}")
    print(f"STEP 1/4: LOADING MODEL")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
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

    model = model.to(device)
    model.eval()
    print(f"✓ Model moved to {device} and set to eval mode")

    # Create validation transforms (deterministic, no augmentation)
    val_transforms = get_contrastive_transforms(
        keys=("vol1", "vol2"),
        input_size=input_size,
        conservative_mode=True,
        val_mode=True,  # No augmentations
        recon=False,
    )

    # Create data module
    print(f"\n{'='*80}")
    print(f"STEP 2/4: LOADING DATASET")
    print(f"{'='*80}")
    print(f"Data directory: {data_dir}")
    print(f"Seed: {seed} (for validation split)")
    data_module = ContrastiveDataModule(
        data_dir=str(data_dir),
        train_transforms=val_transforms,
        val_transforms=val_transforms,
        contrastive_mode="modality_pairs",
        input_size=input_size,
        batch_size=1,
        num_workers=0,
        seed=seed,
    )

    data_module.setup('fit')
    val_loader = data_module.val_dataloader()

    print(f"✓ Validation dataset size: {len(val_loader.dataset)} samples")
    print(f"✓ Validation split: 98% train / 2% validation")

    # Extract embeddings
    total_samples = len(val_loader.dataset)
    print(f"\n{'='*80}")
    print(f"STEP 3/4: EXTRACTING EMBEDDINGS")
    print(f"{'='*80}")
    print(f"Total samples to process: {total_samples}")
    print(f"Batch size: 1")
    print(f"Device: {device}")
    print(f"Embedding dimension: 768")
    print(f"{'='*80}\n")

    all_embeddings = []
    all_filenames = []
    all_modalities = []
    all_patient_ids = []
    all_session_ids = []

    # Access dataset pairs directly to get file paths
    dataset = val_loader.dataset

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Extracting', unit='sample', total=total_samples)):
            # Get volume from batch
            volume = batch['vol1'].to(device)

            # Get metadata from batch
            patient_id = batch.get('patient', f'unknown_{batch_idx}')
            session_id = batch.get('session', 'unknown')

            # Convert to string (handle tensor, list, or string)
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

            # Extract modality from dataset's internal pairs list
            if hasattr(dataset, 'pairs') and batch_idx < len(dataset.pairs):
                pair_info = dataset.pairs[batch_idx]
                path1 = pair_info.get('path1', '')
                if path1:
                    filename = Path(path1).name
                    modality = Path(path1).stem
                    # Remove numeric suffixes like t1_2 -> t1
                    modality = re.sub(r'_\d+$', '', modality)
                else:
                    filename = f'sample_{batch_idx}.npy'
                    modality = 'unknown'
            else:
                filename = f'sample_{batch_idx}.npy'
                modality = 'unknown'

            # Extract features from encoder (before probing layer)
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
            all_modalities.append(modality)
            all_patient_ids.append(patient_id)
            all_session_ids.append(session_id)

            # Milestone logging (every 25%)
            progress = (batch_idx + 1) / total_samples
            if (batch_idx + 1) % max(1, total_samples // 4) == 0:
                print(f"\n[{progress*100:.0f}%] Processed {batch_idx + 1}/{total_samples} samples")
                print(f"  Current modalities seen: {set(all_modalities)}")
                print(f"  Unique patients so far: {len(set(all_patient_ids))}")

    # Concatenate all embeddings into single tensor
    embeddings = torch.cat(all_embeddings, dim=0)  # (N, 768)

    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'filename': all_filenames,
        'embedding_index': range(len(all_filenames)),
        'modality': all_modalities,
        'patient_id': all_patient_ids,
        'session_id': all_session_ids,
    })

    print(f"\nExtraction complete!")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Unique patients: {metadata['patient_id'].nunique()}")
    print(f"  Unique modalities: {metadata['modality'].nunique()}")
    print(f"  Modalities: {sorted(metadata['modality'].unique())}")

    return embeddings, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from pretrained checkpoint"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.ckpt)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to pretrain data directory (hierarchical: sub_X/ses_Y/*.npy)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for embeddings and metadata'
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
        help='Random seed for validation split (default: 42)'
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

    # Validate inputs
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("EMBEDDING EXTRACTION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Input size: {args.input_size}")
    print(f"Seed: {args.seed}")
    print("="*80)

    # Extract embeddings
    embeddings, metadata = extract_embeddings(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
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
