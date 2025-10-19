"""Visualization utilities.

Displays volume slices in three orthogonal planes.
Volume shape: (C, D1, D2, D3) where C is channels and D1, D2, D3 are spatial dimensions.
"""

from __future__ import annotations

__all__ = [
    "plot_npy_volumes",
]

from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import seaborn as sns

def _get_mid_slices(volume: NDArray) -> Tuple[int, int, int]:
    """
    Get mid slice indices for each spatial dimension.
    
    Args:
        volume: 4D volume (C, D1, D2, D3) or 3D volume (D1, D2, D3)
    
    Returns:
        Mid slice indices for (dim1, dim2, dim3)
    """
    if volume.ndim == 4:
        C, D1, D2, D3 = volume.shape
    elif volume.ndim == 3:
        D1, D2, D3 = volume.shape
        C = 1
    else:
        raise ValueError(f"Expected 3D or 4D volume, got {volume.ndim}D")
    
    # Mid slices for each dimension
    mid_d1 = D1 // 2
    mid_d2 = D2 // 2  
    mid_d3 = D3 // 2
    
    return mid_d1, mid_d2, mid_d3

def plot_npy_volumes(
    volumes: Dict[str, NDArray],
    slice_indices: Optional[Tuple[int, int, int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12),
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot multiple volumes in a single figure for comparison.
    
    Args:
        volumes: Dict mapping volume names to volume arrays (C, D1, D2, D3)
        slice_indices: Custom slice indices (dim1, dim2, dim3)
        titles: Column titles for each plane (default: ["Dim2-Dim3", "Dim1-Dim3", "Dim1-Dim2"])
        figsize: Figure size
        cmap: Colormap
        vmin, vmax: Value range for colormap
        save_path: Path to save figure (optional)
    """
    volume_names = list(volumes.keys())
    n_volumes = len(volume_names)
    
    # Default titles for the three orthogonal planes
    # Note: imshow displays first array dim as y-axis, second as x-axis
    sub_titles = ["Dim3-Dim2", "Dim3-Dim1", "Dim2-Dim1"]
    main_title = title or "Multi-Volume Comparison - Three Orthogonal Planes"
    
    # Create figure with subplots (n_volumes rows, 3 columns)
    fig, axes = plt.subplots(n_volumes, 3, figsize=figsize)
    
    # Handle single volume case
    if n_volumes == 1:
        axes = axes.reshape(1, -1)
    
    # Get slice indices (use first volume to determine dimensions)
    first_volume = volumes[volume_names[0]]
    if slice_indices is None:
        slice_indices = _get_mid_slices(first_volume)
    
    idx_d1, idx_d2, idx_d3 = slice_indices
    
    # Plot each volume
    for row_idx, vol_name in enumerate(volume_names):
        volume = volumes[vol_name]
        
        # Handle channel dimension - extract (D1, D2, D3) from (C, D1, D2, D3)
        if volume.ndim == 4:
            volume = volume[0]
        
        # Plane 1: Slice through dim1, shows dim2-dim3 plane
        # volume[idx_d1, :, :] gives us (D2, D3)
        # imshow: first array dim -> y-axis, second array dim -> x-axis
        im1 = axes[row_idx, 0].imshow(
            volume[idx_d1, :, :], 
            cmap=cmap, 
            origin='lower',
            vmin=vmin, 
            vmax=vmax,
            aspect='equal'
        )
        axes[row_idx, 0].set_title(f"{vol_name}\n{sub_titles[0]} (D1={idx_d1})", fontsize=10, fontweight='bold')
        axes[row_idx, 0].set_xlabel("Dim3")
        axes[row_idx, 0].set_ylabel("Dim2")
        axes[row_idx, 0].grid(False)
        plt.colorbar(im1, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)
        
        # Plane 2: Slice through dim2, shows dim1-dim3 plane
        # volume[:, idx_d2, :] gives us (D1, D3)
        # imshow: first array dim -> y-axis, second array dim -> x-axis
        im2 = axes[row_idx, 1].imshow(
            volume[:, idx_d2, :], 
            cmap=cmap, 
            origin='lower',
            vmin=vmin, 
            vmax=vmax,
            aspect='equal'
        )
        axes[row_idx, 1].set_title(f"{vol_name}\n{sub_titles[1]} (D2={idx_d2})", fontsize=10, fontweight='bold')
        axes[row_idx, 1].set_xlabel("Dim3")
        axes[row_idx, 1].set_ylabel("Dim1")
        axes[row_idx, 1].grid(False)
        plt.colorbar(im2, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
        
        # Plane 3: Slice through dim3, shows dim1-dim2 plane
        # volume[:, :, idx_d3] gives us (D1, D2)
        # imshow: first array dim -> y-axis, second array dim -> x-axis
        im3 = axes[row_idx, 2].imshow(
            volume[:, :, idx_d3], 
            cmap=cmap, 
            origin='lower',
            vmin=vmin, 
            vmax=vmax,
            aspect='equal'
        )
        axes[row_idx, 2].set_title(f"{vol_name}\n{sub_titles[2]} (D3={idx_d3})", fontsize=10, fontweight='bold')
        axes[row_idx, 2].set_xlabel("Dim2")
        axes[row_idx, 2].set_ylabel("Dim1")
        axes[row_idx, 2].grid(False)
        plt.colorbar(im3, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle(main_title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()