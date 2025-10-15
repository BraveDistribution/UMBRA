"""
Augmentation utilities for self-supervised learning.

This package provides data augmentation functions for medical imaging,
particularly focused on masking strategies for masked autoencoding.
"""

from .composed import get_mae_transforms, get_contrastive_transforms

__all__ = ["get_mae_transforms", "get_contrastive_transforms"]
