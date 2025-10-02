"""
Visualization script for demonstrating modality-agnostic feature learning.

This script extracts embeddings from different medical imaging modalities (CT, MRI, PET, etc.)
and creates t-SNE visualizations to show that the foundation model learns modality-agnostic
representations where similar anatomical structures cluster together regardless of scan type.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import re
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.foundation import ContrastiveTransformer


def center_crop(
    volume: np.ndarray,
    crop_size: Tuple[int, int, int] = (96, 96, 96),
) -> np.ndarray:
    """
    Center crop a volume to the specified size, matching training preprocessing.

    Args:
        volume: Input volume of shape (C, D, H, W) or (D, H, W)
        crop_size: Target crop size (depth, height, width)

    Returns:
        Cropped volume
    """
    # Ensure 4D shape (C, D, H, W)
    if volume.ndim == 3:
        volume = volume[np.newaxis, ...]

    _C, D, H, W = volume.shape
    cd, ch, cw = crop_size

    # Calculate center crop coordinates
    start_d = max(0, (D - cd) // 2)
    start_h = max(0, (H - ch) // 2)
    start_w = max(0, (W - cw) // 2)

    # Crop
    cropped = volume[:, start_d:start_d + cd, start_h:start_h + ch, start_w:start_w + cw]

    # Pad if volume is smaller than crop_size
    pad_d = max(0, cd - cropped.shape[1])
    pad_h = max(0, ch - cropped.shape[2])
    pad_w = max(0, cw - cropped.shape[3])

    if pad_d or pad_h or pad_w:
        cropped = np.pad(
            cropped,
            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0,
        )

    return cropped


def get_patient_split(
    data_dir: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    Get train/val patient split using the same logic as ContrastiveDataModule.

    This matches the exact split logic from data/contrastive.py to ensure
    validation embeddings correspond to the actual validation set.

    Args:
        data_dir: Directory containing .npy files
        test_size: Fraction of patients for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        train_patients: Set of training patient IDs (as strings)
        val_patients: Set of validation patient IDs (as strings)
    """
    data_dir = Path(data_dir)
    SESSION_RE = re.compile(
        r"sub_(?P<patient>\d+)_ses_(?P<session>\d+)_(?P<scan_type>.+)\.npy"
    )

    # Extract unique patient IDs
    patient_ids = {
        int(match.group("patient"))
        for filename in os.listdir(data_dir)
        if (match := SESSION_RE.match(filename))
    }

    # Convert to strings and split (same as ContrastiveDataModule)
    patient_ids_str = [str(i) for i in patient_ids]
    train_patients_list, val_patients_list = train_test_split(
        patient_ids_str, test_size=test_size, random_state=random_state
    )

    return set(train_patients_list), set(val_patients_list)


def extract_embeddings_from_checkpoint(
    checkpoint_path: Union[str, Path],
    data_dir: Union[str, Path],
    max_samples_per_modality: Optional[int] = 100,
    split: str = "val",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract embeddings from pretrained model for different modalities.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing .npy files with naming convention:
                  sub_{patient}_ses_{session}_{scan_type}.npy
        max_samples_per_modality: Maximum number of samples to extract per modality
        split: Which split to use ('train', 'val', or 'all')
        device: Device to run inference on

    Returns:
        embeddings: Array of shape (N, embedding_dim)
        modalities: List of modality labels for each embedding
        patient_ids: List of patient IDs for each embedding
    """
    print(f"Loading model from {checkpoint_path}...")
    model = ContrastiveTransformer.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    # Get patient split using the same logic as ContrastiveDataModule
    train_patients, val_patients = get_patient_split(data_dir)

    if split == "train":
        allowed_patients = train_patients
        print(f"\nUsing TRAIN split: {len(train_patients)} patients")
    elif split == "val":
        allowed_patients = val_patients
        print(f"\nUsing VALIDATION split: {len(val_patients)} patients")
    elif split == "all":
        allowed_patients = train_patients | val_patients
        print(f"\nUsing ALL data: {len(allowed_patients)} patients")
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'all'")

    # Parse data directory to find all files and group by modality
    data_dir = Path(data_dir)
    SESSION_RE = re.compile(
        r"sub_(?P<patient>\d+)_ses_(?P<session>\d+)_(?P<scan_type>.+)\.npy"
    )

    # Group files by modality (filtered by split)
    modality_files: Dict[str, List[Tuple[str, str, Path]]] = {}
    for file_path in data_dir.glob("*.npy"):
        match = SESSION_RE.match(file_path.name)
        if match:
            patient = match.group("patient")
            scan_type = match.group("scan_type")

            # Only include files from the selected split
            if patient not in allowed_patients:
                continue

            if scan_type not in modality_files:
                modality_files[scan_type] = []
            modality_files[scan_type].append((patient, scan_type, file_path))

    print(f"\nFound modalities: {list(modality_files.keys())}")
    for modality, files in modality_files.items():
        print(f"  {modality}: {len(files)} files")

    # Extract embeddings for each modality
    all_embeddings = []
    all_modalities = []
    all_patients = []

    with torch.no_grad():
        for modality, files in modality_files.items():
            print(f"\nProcessing {modality}...")

            # Limit samples per modality if specified
            if max_samples_per_modality is not None:
                files = files[:max_samples_per_modality]

            for patient, scan_type, file_path in tqdm(
                files, desc=f"Extracting {modality}"
            ):
                try:
                    # Load volume
                    volume = np.load(file_path)

                    # Handle NaN/Inf values (same as contrastive_dataset.py)
                    if np.isnan(volume).any() or np.isinf(volume).any():
                        volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0, copy=True)

                    # Ensure 4D shape (C, D, H, W) and center crop to 96x96x96
                    volume = center_crop(volume, crop_size=(96, 96, 96))

                    # Convert to tensor and add batch dimension
                    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)  # [1, C, D, H, W]
                    volume_tensor = volume_tensor.to(device)

                    # Extract features using the contrastive projection
                    embedding = model.forward_contrastive(volume_tensor)

                    # Convert to numpy and store
                    embedding_np = embedding.cpu().numpy().squeeze()
                    all_embeddings.append(embedding_np)
                    all_modalities.append(scan_type)
                    all_patients.append(patient)

                except Exception as e:
                    print(f"  Warning: Failed to process {file_path.name}: {e}")
                    continue

    embeddings = np.array(all_embeddings)
    print(f"\nExtracted {len(embeddings)} embeddings with shape {embeddings.shape}")

    return embeddings, all_modalities, all_patients


def create_tsne_visualization(
    embeddings: np.ndarray,
    modalities: List[str],
    patient_ids: List[str],
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Create t-SNE projection of embeddings.

    Args:
        embeddings: Array of embeddings
        modalities: List of modality labels
        patient_ids: List of patient IDs
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed

    Returns:
        tsne_embeddings: 2D t-SNE coordinates
    """
    print(f"\nRunning t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1,
    )

    tsne_embeddings = tsne.fit_transform(embeddings)
    print("t-SNE completed!")

    return tsne_embeddings


def plot_modality_visualization(
    tsne_embeddings: np.ndarray,
    modalities: List[str],
    patient_ids: List[str],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Create visualization showing modality-agnostic clustering.

    Args:
        tsne_embeddings: 2D t-SNE coordinates
        modalities: List of modality labels
        patient_ids: List of patient IDs
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)

    # Get unique modalities and assign colors
    unique_modalities = sorted(list(set(modalities)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_modalities)))
    modality_to_color = dict(zip(unique_modalities, colors))

    # Create scatter plot
    for modality in unique_modalities:
        mask = np.array(modalities) == modality
        plt.scatter(
            tsne_embeddings[mask, 0],
            tsne_embeddings[mask, 1],
            c=[modality_to_color[modality]],
            label=modality,
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )

    plt.xlabel("t-SNE Component 1", fontsize=14, fontweight="bold")
    plt.ylabel("t-SNE Component 2", fontsize=14, fontweight="bold")
    plt.title(
        "Modality-Agnostic Feature Representation\n"
        "Different colors = Different modalities",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.legend(
        title="Modality",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=11,
        title_fontsize=12,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved visualization to {save_path}")

    plt.show()


def plot_patient_visualization(
    tsne_embeddings: np.ndarray,
    modalities: List[str],
    patient_ids: List[str],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
    max_patients_to_show: int = 10,
) -> None:
    """
    Create visualization colored by patient ID to show patient-specific clustering.

    Args:
        tsne_embeddings: 2D t-SNE coordinates
        modalities: List of modality labels
        patient_ids: List of patient IDs
        save_path: Path to save figure (optional)
        figsize: Figure size
        max_patients_to_show: Maximum number of patients to color distinctly
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)

    # Get unique patients
    unique_patients = sorted(list(set(patient_ids)))[:max_patients_to_show]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_patients)))
    patient_to_color = dict(zip(unique_patients, colors))

    # Plot patients with distinct colors, others in gray
    for patient in unique_patients:
        mask = np.array(patient_ids) == patient
        plt.scatter(
            tsne_embeddings[mask, 0],
            tsne_embeddings[mask, 1],
            c=[patient_to_color[patient]],
            label=f"Patient {patient}",
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )

    # Plot remaining patients in gray
    other_mask = ~np.isin(patient_ids, unique_patients)
    if other_mask.any():
        plt.scatter(
            tsne_embeddings[other_mask, 0],
            tsne_embeddings[other_mask, 1],
            c="lightgray",
            label="Other patients",
            alpha=0.3,
            s=30,
        )

    plt.xlabel("t-SNE Component 1", fontsize=14, fontweight="bold")
    plt.ylabel("t-SNE Component 2", fontsize=14, fontweight="bold")
    plt.title(
        "Patient-Specific Clustering\nDifferent colors = Different patients",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.legend(
        title="Patient ID",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
        title_fontsize=12,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved visualization to {save_path}")

    plt.show()


def analyze_modality_mixing(
    embeddings: np.ndarray,
    modalities: List[str],
) -> None:
    """
    Compute and display statistics about modality mixing in the embedding space.

    Args:
        embeddings: Array of embeddings
        modalities: List of modality labels
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    print("\n" + "=" * 60)
    print("MODALITY MIXING ANALYSIS")
    print("=" * 60)

    # Encode modalities as integers
    le = LabelEncoder()
    modality_labels = le.fit_transform(modalities)

    # Compute silhouette score (lower = more mixing = better modality-agnostic)
    # Score ranges from -1 to 1:
    # - Close to 1: modalities are well-separated (bad for modality-agnostic)
    # - Close to 0: modalities are mixed (good for modality-agnostic)
    # - Close to -1: samples might be assigned to wrong clusters
    silhouette = silhouette_score(embeddings, modality_labels)

    print(f"\nSilhouette Score: {silhouette:.4f}")
    print("  (Lower score indicates better modality mixing)")
    print("  - Close to 0: Excellent modality-agnostic learning")
    print("  - 0.3-0.5: Moderate modality mixing")
    print("  - >0.5: Poor modality mixing (modality-specific features)")

    # Compute pairwise distances within and between modalities
    from scipy.spatial.distance import cdist

    unique_modalities = sorted(list(set(modalities)))
    print(f"\nAverage pairwise distances:")

    for i, mod_i in enumerate(unique_modalities):
        mask_i = np.array(modalities) == mod_i
        emb_i = embeddings[mask_i]

        # Within-modality distance
        if len(emb_i) > 1:
            within_dist = cdist(emb_i, emb_i).mean()
            print(f"  Within {mod_i}: {within_dist:.4f}")

        # Between-modality distances
        for mod_j in unique_modalities[i + 1 :]:
            mask_j = np.array(modalities) == mod_j
            emb_j = embeddings[mask_j]
            between_dist = cdist(emb_i, emb_j).mean()
            print(f"  Between {mod_i} and {mod_j}: {between_dist:.4f}")

    print("\n" + "=" * 60)


def main(
    checkpoint_path: Union[str, Path],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path] = "visualizations",
    max_samples_per_modality: int = 100,
    split: str = "val",
    perplexity: int = 30,
    n_iter: int = 1000,
) -> None:
    """
    Main function to generate all visualizations.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing .npy files
        output_dir: Directory to save visualizations
        max_samples_per_modality: Maximum samples per modality
        split: Which split to use ('train', 'val', or 'all')
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
    """
    # Create output directory
    output_dir_path: Path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    # Extract embeddings
    embeddings, modalities, patient_ids = extract_embeddings_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        max_samples_per_modality=max_samples_per_modality,
        split=split,
    )

    # Save embeddings for future use
    np.save(output_dir_path / "embeddings.npy", embeddings)
    np.save(output_dir_path / "modalities.npy", modalities)
    np.save(output_dir_path / "patient_ids.npy", patient_ids)
    print(f"\nSaved embeddings to {output_dir_path}")

    # Analyze modality mixing
    analyze_modality_mixing(embeddings, modalities)

    # Create t-SNE visualization
    tsne_embeddings = create_tsne_visualization(
        embeddings=embeddings,
        modalities=modalities,
        patient_ids=patient_ids,
        perplexity=perplexity,
        n_iter=n_iter,
    )

    # Save t-SNE coordinates
    tsne_path: Path = output_dir_path / "tsne_embeddings.npy"
    np.save(tsne_path, tsne_embeddings)

    # Create visualizations
    modality_plot_path: Path = output_dir_path / "tsne_by_modality.png"
    plot_modality_visualization(
        tsne_embeddings=tsne_embeddings,
        modalities=modalities,
        patient_ids=patient_ids,
        save_path=modality_plot_path,
    )

    patient_plot_path: Path = output_dir_path / "tsne_by_patient.png"
    plot_patient_visualization(
        tsne_embeddings=tsne_embeddings,
        modalities=modalities,
        patient_ids=patient_ids,
        save_path=patient_plot_path,
    )

    print("\n✓ All visualizations completed!")
    print(f"Results saved in: {output_dir_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize modality-agnostic embeddings from pretrained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing .npy files with format: sub_{patient}_ses_{session}_{scan_type}.npy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations (default: visualizations/)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum samples per modality (default: 100)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "all"],
        help="Which split to use: 'train', 'val', or 'all' (default: val)",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter (default: 30)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1000,
        help="Number of t-SNE iterations (default: 1000)",
    )

    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples_per_modality=args.max_samples,
        split=args.split,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
    )
