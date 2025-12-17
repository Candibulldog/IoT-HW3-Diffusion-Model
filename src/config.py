# src/config.py

"""Configuration module for the DDPM project.

This module defines the `Config` class, which serves as a centralized container
for all hyperparameters, file paths, and device settings used throughout
the training and generation pipelines.
"""

from pathlib import Path

import torch


class Config:
    """Central configuration for the diffusion model.

    Attributes:
        DATA_PATH (Path): Path to the directory containing the dataset.
        IMAGE_SIZE (int): Height and width of the input images (square).
        IMAGE_CHANNELS (int): Number of channels in the input/output images.
                              Set to 1 for MNIST (Grayscale) for efficiency,
                              though generated images are converted to RGB for submission.

        MODEL_CHANNELS (int): Base number of channels in the U-Net.
        CHANNEL_MULT (tuple): Multipliers for channels at each U-Net resolution level.
        NUM_RES_BLOCKS (int): Number of residual blocks per resolution level.
        ATTENTION_RESOLUTIONS (tuple): Resolutions at which self-attention is applied.
        NUM_HEADS (int): Number of attention heads.
        DROPOUT (float): Dropout rate used in ResBlocks.
        TIME_EMB_DIM (int): Dimension of the time embedding vector.

        TIMESTEPS (int): Total number of diffusion steps (T).
        BETA_SCHEDULE (str): Type of noise schedule ('linear' or 'cosine').
        BETA_START (float): Starting value of beta.
        BETA_END (float): Ending value of beta.
        COSINE_S (float): Offset 's' for cosine schedule to prevent singularity.

        EPOCHS (int): Total number of training epochs.
        BATCH_SIZE (int): Batch size for training.
        LEARNING_RATE (float): Initial learning rate for the optimizer.
        WEIGHT_DECAY (float): Weight decay (L2 penalty) for the optimizer.
        GRAD_CLIP (float): Maximum norm for gradient clipping.

        USE_EMA (bool): Whether to use Exponential Moving Average for model weights.
        EMA_DECAY (float): Decay rate for EMA.
        EMA_UPDATE_EVERY (int): Frequency (in steps) to update EMA weights.

        USE_LR_SCHEDULER (bool): Whether to use a learning rate scheduler.
        LR_SCHEDULER_TYPE (str): Type of scheduler (e.g., 'cosine').
        LR_MIN (float): Minimum learning rate for cosine annealing.

        USE_AUGMENTATION (bool): Whether to apply data augmentation during training.
        ROTATION_DEGREES (float): Max degrees for random rotation augmentation.

        SAMPLING_METHOD (str): Default sampling method ('ddpm' or 'ddim').
        DDIM_SAMPLING_STEPS (int): Number of steps for DDIM sampling.
        DDIM_ETA (float): Eta parameter for DDIM (0.0 for deterministic).
        NUM_GENERATED_IMAGES (int): Total number of images to generate for submission.
        GENERATION_BATCH_SIZE (int): Batch size during generation.

        SAVE_IMAGE_PATH (Path): Directory to save generated images.
        CHECKPOINT_DIR (Path): Directory to save model checkpoints.
        DEVICE (str): Computation device ('cuda' or 'cpu').
    """

    # ========================
    # Data Settings
    # ========================
    DATA_PATH: Path = Path("./mnist")
    IMAGE_SIZE: int = 28
    IMAGE_CHANNELS: int = 1

    # ========================
    # Model Architecture
    # ========================
    MODEL_CHANNELS: int = 128
    CHANNEL_MULT: tuple[int, ...] = (1, 2, 4)
    NUM_RES_BLOCKS: int = 2
    ATTENTION_RESOLUTIONS: tuple[int, ...] = (7,)
    NUM_HEADS: int = 8
    DROPOUT: float = 0.1
    TIME_EMB_DIM: int = 512

    # ========================
    # Diffusion Settings
    # ========================
    TIMESTEPS: int = 1000
    BETA_SCHEDULE: str = "cosine"
    BETA_START: float = 0.0001
    BETA_END: float = 0.02
    COSINE_S: float = 0.008

    # ========================
    # Training Settings
    # ========================
    EPOCHS: int = 120
    BATCH_SIZE: int = 384
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 0.0
    GRAD_CLIP: float = 1.0

    # EMA settings
    USE_EMA: bool = True
    EMA_DECAY: float = 0.995
    EMA_UPDATE_EVERY: int = 10

    # Learning rate scheduler
    USE_LR_SCHEDULER: bool = True
    LR_SCHEDULER_TYPE: str = "cosine"
    LR_MIN: float = 1e-6

    # Data augmentation
    USE_AUGMENTATION: bool = True
    ROTATION_DEGREES: float = 5.0

    # ========================
    # Generation Settings
    # ========================
    SAMPLING_METHOD: str = "ddim"
    DDIM_SAMPLING_STEPS: int = 50
    DDIM_ETA: float = 0.0
    NUM_GENERATED_IMAGES: int = 10000
    GENERATION_BATCH_SIZE: int = 64

    # ========================
    # Paths
    # ========================
    SAVE_IMAGE_PATH: Path = Path("./generated_images")
    CHECKPOINT_DIR: Path = Path("./checkpoints")

    # ========================
    # Device Settings
    # ========================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 8
    PIN_MEMORY: bool = True

    # ========================
    # Logging & Visualization
    # ========================
    CHECKPOINT_EVERY: int = 10
    SAVE_SAMPLE_IMAGES: bool = True
    SAMPLE_EVERY: int = 10
    NUM_SAMPLE_IMAGES: int = 64
    LOG_INTERVAL: int = 100

    @classmethod
    def print_config(cls):
        """Prints the configuration settings in a readable format to stdout."""
        header = " ⚙️ CONFIGURATION ⚙️ "
        line_length = 60
        padding = (line_length - len(header)) // 2

        print("\n" + "=" * padding + header + "=" * padding)
        config_vars = {k: v for k, v in cls.__dict__.items() if not k.startswith("__") and not callable(v)}
        for key, value in config_vars.items():
            print(f"  {key:25s}: {value}")
        print("=" * line_length + "\n")


if __name__ == "__main__":
    Config.print_config()
