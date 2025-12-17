# src/generate.py

"""Image Generation Script for HW3 Submission.

This script loads a trained Diffusion Model and generates 10,000 images
strictly following the homework specifications:
1. Filenames: 00001.png to 10000.png
2. Format: PNG
3. Color: RGB (converted from Grayscale)
4. Directory: No subdirectories
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.config import Config
from src.diffusion import DiffusionScheduler
from src.model import UNet


class ImageGenerator:
    """Orchestrates the model loading and image generation process."""

    def __init__(self, model_path: Path, config: Config, use_ema: bool):
        """Initialize the generator.

        Args:
            model_path (Path): Path to the .pth checkpoint file.
            config (Config): Configuration object.
            use_ema (bool): Whether to load EMA weights if available.
        """
        self.config = config
        self.device = config.DEVICE

        self.model = self._load_model(model_path, use_ema)
        self.diffusion = self._init_diffusion()

        print("\n" + "---" * 5 + " ✨ GENERATOR INITIALIZED ✨ " + "---" * 5)
        print(f"  Device:         {self.device}")
        print(f"  Model loaded:   '{model_path}'")
        print(f"  Using EMA:      {use_ema}")
        print("---" * 17 + "\n")

    def _load_model(self, model_path: Path, use_ema: bool) -> torch.nn.Module:
        """Loads the U-Net from disk."""
        model_config = {
            "in_channels": self.config.IMAGE_CHANNELS,
            "out_channels": self.config.IMAGE_CHANNELS,
            "model_channels": self.config.MODEL_CHANNELS,
            "channel_mult": self.config.CHANNEL_MULT,
            "num_res_blocks": self.config.NUM_RES_BLOCKS,
            "attention_resolutions": self.config.ATTENTION_RESOLUTIONS,
            "num_heads": self.config.NUM_HEADS,
            "dropout": 0.0,  # Dropout is irrelevant for inference
            "time_emb_dim": self.config.TIME_EMB_DIM,
            "image_size": self.config.IMAGE_SIZE,
        }
        model = UNet(**model_config)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Logic to load EMA weights if requested and present
        if use_ema and "ema_state_dict" in checkpoint:
            print("  -> Found and loading EMA weights.")
            model.load_state_dict(checkpoint["ema_state_dict"])
        elif "model_state_dict" in checkpoint:
            print("  -> EMA weights not found or not requested. Loading standard model weights.")
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("  -> Assuming the file is a raw state_dict. Loading directly.")
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def _init_diffusion(self) -> DiffusionScheduler:
        """Sets up the diffusion noise scheduler."""
        return DiffusionScheduler(
            timesteps=self.config.TIMESTEPS,
            beta_schedule=self.config.BETA_SCHEDULE,
            beta_start=self.config.BETA_START,
            beta_end=self.config.BETA_END,
            cosine_s=self.config.COSINE_S,
            device=self.device,
        )

    @torch.no_grad()
    def generate(self, num_images: int, batch_size: int, sampling_method: str, sampling_steps: int) -> torch.Tensor:
        """Runs the full reverse diffusion loop to generate images.

        Args:
            num_images (int): Total images to generate.
            batch_size (int): Batch size per inference step.
            sampling_method (str): 'ddpm' or 'ddim'.
            sampling_steps (int): Number of steps for DDIM.

        Returns:
            torch.Tensor: Generated images tensor (N, C, H, W), values in [0, 1].
        """
        all_images = []
        num_batches = (num_images + batch_size - 1) // batch_size

        for _ in tqdm(range(num_batches), desc="🎨 Generating images"):
            current_batch_size = min(batch_size, num_images - len(all_images))
            if current_batch_size <= 0:
                break

            shape = (current_batch_size, self.config.IMAGE_CHANNELS, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)

            images = self.diffusion.sample(
                model=self.model,
                shape=shape,
                method=sampling_method,
                sampling_steps=sampling_steps,
                eta=self.config.DDIM_ETA,
            )

            # Denormalize: [-1, 1] -> [0, 1]
            images = (images.clamp(-1, 1) + 1) / 2
            all_images.append(images.cpu())

        generated_tensor = torch.cat(all_images, dim=0)
        return generated_tensor[:num_images]


def save_image_batch(images: torch.Tensor, output_dir: Path):
    """Saves images to disk in the required format.

    Args:
        images (torch.Tensor): Tensor of images (N, C, H, W).
        output_dir (Path): Destination folder.

    Note:
        Converts 1-channel images to 3-channel RGB to satisfy FID calculation requirements.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # CRITICAL: Ensure 3 channels for FID compatibility
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    for i, img in enumerate(tqdm(images, desc="💾 Saving images")):
        # Filename requirement: 00001.png ... 10000.png
        filename = f"{i + 1:05d}.png"
        save_image(img, output_dir / filename)


def main():
    """Main entry point for generation."""
    parser = argparse.ArgumentParser(
        description="Generate images from a trained Diffusion Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to save generated images.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to generate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for generation.")
    parser.add_argument("--sampling_method", type=str, default=None, choices=["ddpm", "ddim"], help="Sampling method.")
    parser.add_argument("--sampling_steps", type=int, default=None, help="Number of sampling steps (for DDIM).")
    parser.add_argument("--no_ema", action="store_true", help="Do not use EMA weights, even if they are available.")

    args = parser.parse_args()

    # Update config with CLI args
    config = Config()
    if args.output_dir is not None:
        config.SAVE_IMAGE_PATH = args.output_dir
    if args.num_images is not None:
        config.NUM_GENERATED_IMAGES = args.num_images
    if args.batch_size is not None:
        config.GENERATION_BATCH_SIZE = args.batch_size
    if args.sampling_method is not None:
        config.SAMPLING_METHOD = args.sampling_method
    if args.sampling_steps is not None:
        config.DDIM_SAMPLING_STEPS = args.sampling_steps

    generator = ImageGenerator(model_path=args.model_path, config=config, use_ema=not args.no_ema)

    print(f"\nGenerating {config.NUM_GENERATED_IMAGES} images for submission...")
    images_to_save = generator.generate(
        num_images=config.NUM_GENERATED_IMAGES,
        batch_size=config.GENERATION_BATCH_SIZE,
        sampling_method=config.SAMPLING_METHOD,
        sampling_steps=config.DDIM_SAMPLING_STEPS,
    )

    save_image_batch(images_to_save, config.SAVE_IMAGE_PATH)

    print("\n" + "---" * 10 + " 🎉 GENERATION COMPLETE 🎉 " + "---" * 10)
    print(f"  Successfully generated and saved {len(images_to_save)} images.")
    print(f"  Output directory: '{config.SAVE_IMAGE_PATH}'")
    print("---" * 27 + "\n")


if __name__ == "__main__":
    main()
