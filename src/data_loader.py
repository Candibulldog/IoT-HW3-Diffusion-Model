# src/data_loader.py

"""Data loading and augmentation pipeline.

This module handles the preparation of the MNIST dataset.
It includes a custom Dataset class for loading PNG images and a
function to create efficient DataLoaders.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    """Custom Dataset for loading MNIST images from a local directory.

    Handles reading PNG files, converting them to tensors, and applying
    transformations (resizing, rotation, normalization).
    """

    def __init__(
        self, root_dir: Path, image_size: int, use_augmentation: bool, augmentation_params: dict[str, float] | None
    ):
        """Initializes the dataset.

        Args:
            root_dir (Path): Path to the folder containing .png images.
            image_size (int): Target height/width for resizing.
            use_augmentation (bool): Whether to apply random transformations.
            augmentation_params (dict): Parameters for augmentation (e.g., {'rotation': 5.0}).
        """
        self.image_paths = sorted(root_dir.glob("*.png"))

        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in the directory: {root_dir}")

        self.transform = self._build_transforms(image_size, use_augmentation, augmentation_params)

    def _build_transforms(self, image_size: int, use_augmentation: bool, aug_params: dict[str, float] | None):
        """Constructs the composition of transforms."""
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]

        if use_augmentation:
            params = aug_params or {"rotation": 5.0}
            rotation_deg = params.get("rotation", 0.0)
            if rotation_deg > 0:
                transform_list.append(transforms.RandomRotation(degrees=rotation_deg))

        # Normalization: [0, 1] -> [-1, 1]
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Retrieves an image by index.

        Returns:
            tuple: (image_tensor, label). Label is always 0 (unsupervised).
        """
        img_path = self.image_paths[idx]
        # Load as Grayscale (L) to match model input channels (1)
        image = Image.open(img_path).convert("L")
        return self.transform(image), 0


def get_dataloader(
    data_path: Path,
    batch_size: int,
    image_size: int,
    use_augmentation: bool,
    augmentation_params: dict[str, float] | None,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool = True,
) -> DataLoader:
    """Creates a PyTorch DataLoader for the dataset.

    Args:
        data_path (Path): Path to the image directory.
        batch_size (int): Number of samples per batch.
        image_size (int): Image resolution.
        use_augmentation (bool): Enable augmentation.
        augmentation_params (dict): Augmentation settings.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): Copy tensors into CUDA pinned memory.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Configured data loader.
    """
    dataset = MNISTDataset(
        root_dir=data_path,
        image_size=image_size,
        use_augmentation=use_augmentation,
        augmentation_params=augmentation_params,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
