"""
Augmentation utilities for self-supervised learning.

This package provides data augmentation functions for medical imaging,
particularly focused on masking strategies for masked autoencoding.
"""

from augmentations.mask import random_mask

__all__ = ["random_mask"]
