# src/utils.py

"""Utility functions for visualization and analysis.

This module contains helper functions to visualize generated samples
and compare different diffusion noise schedules. These utilities are
useful for monitoring training progress and generating figures for the report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def visualize_samples(images: torch.Tensor, save_path: Path | None = None, title: str = ""):
    """Creates and optionally saves a grid of sample images.

    Args:
        images (torch.Tensor): Batch of images (B, C, H, W). Values should be in [0, 1].
        save_path (Path, optional): Path to save the plot. If None, shows the plot.
        title (str, optional): Title for the plot.
    """
    # Calculate grid dimensions
    nrow = int(np.sqrt(images.shape[0]))

    # Create a grid tensor (C, H_grid, W_grid)
    grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)

    # Convert to numpy: (H_grid, W_grid, C)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))

    # Handle Grayscale vs RGB visualization
    if grid_np.shape[-1] == 1:
        # Squeeze last dim for grayscale: (H, W, 1) -> (H, W)
        plt.imshow(grid_np.squeeze(-1), cmap="gray")
    else:
        plt.imshow(grid_np)

    plt.axis("off")
    if title:
        plt.title(title, fontsize=16)

    if save_path:
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📈 Visualization saved to '{save_path}'")
    else:
        plt.show()
    plt.close()
