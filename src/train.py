# src/train.py

"""Main Training Script.

This script orchestrates the entire training pipeline for the Diffusion Model.
It handles:
1. Data loading (MNIST).
2. Model initialization (U-Net with EMA).
3. Optimizer and Scheduler setup (AdamW + Cosine Annealing).
4. The main training loop with gradient clipping and logging.
5. Periodic checkpointing and sample generation.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from src.config import Config
from src.data_loader import get_dataloader
from src.diffusion import DiffusionScheduler
from src.ema import EMA
from src.experiment_logger import ExperimentLogger
from src.model import UNet, count_parameters


class Trainer:
    """Manages the training lifecycle of the diffusion model."""

    def __init__(self, config: Config, experiment_name: str | None = None):
        """Initializes the Trainer.

        Args:
            config (Config): Configuration object.
            experiment_name (str, optional): Custom name for the run folder.
        """
        self.config = config
        self.device = config.DEVICE

        # Setup experiment directory
        if not experiment_name:
            experiment_name = f"DDPM_MNIST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run_dir = Path("./experiments") / experiment_name
        self.logger = ExperimentLogger(log_dir=self.run_dir)
        self.logger.save_config(self.config)

        # Initialize Components
        self.dataloader = self._init_data()
        self.model = self._init_model()
        self.optimizer, self.scheduler = self._init_optimizer_and_scheduler()

        self.diffusion = DiffusionScheduler(
            timesteps=self.config.TIMESTEPS,
            beta_schedule=self.config.BETA_SCHEDULE,
            beta_start=self.config.BETA_START,
            beta_end=self.config.BETA_END,
            cosine_s=self.config.COSINE_S,
            device=self.device,
        )

        # Setup EMA
        if self.config.USE_EMA:
            print("  ✅ EMA enabled.")
            self.ema = EMA(model=self.model, decay=self.config.EMA_DECAY, update_every=self.config.EMA_UPDATE_EVERY)
        else:
            self.ema = None

        self.global_step = 0
        self.current_epoch = 0

        # Print Summary
        print(f"  Model Parameters:     {count_parameters(self.model):,}")
        print(f"  Training Samples:     {len(self.dataloader.dataset):,}")
        print(f"  Batches per Epoch:    {len(self.dataloader):,}")
        print("---------------------------------\n")

    def _init_data(self) -> torch.utils.data.DataLoader:
        """Sets up the DataLoader based on config."""
        aug_params = None
        if self.config.USE_AUGMENTATION:
            # Note: Horizontal flip is disabled in Config/DataLoader
            aug_params = {"rotation": self.config.ROTATION_DEGREES}

        return get_dataloader(
            data_path=Path(self.config.DATA_PATH),
            batch_size=self.config.BATCH_SIZE,
            image_size=self.config.IMAGE_SIZE,
            use_augmentation=self.config.USE_AUGMENTATION,
            augmentation_params=aug_params,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            shuffle=True,
        )

    def _init_model(self) -> torch.nn.Module:
        """Initializes the U-Net."""
        model_config = {
            "in_channels": self.config.IMAGE_CHANNELS,
            "out_channels": self.config.IMAGE_CHANNELS,
            "model_channels": self.config.MODEL_CHANNELS,
            "channel_mult": self.config.CHANNEL_MULT,
            "num_res_blocks": self.config.NUM_RES_BLOCKS,
            "attention_resolutions": self.config.ATTENTION_RESOLUTIONS,
            "num_heads": self.config.NUM_HEADS,
            "dropout": self.config.DROPOUT,
            "time_emb_dim": self.config.TIME_EMB_DIM,
            "image_size": self.config.IMAGE_SIZE,
        }
        return UNet(**model_config).to(self.device)

    def _init_optimizer_and_scheduler(self):
        """Initializes AdamW and Cosine Annealing Scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY
        )

        scheduler = None
        if self.config.USE_LR_SCHEDULER:
            if self.config.LR_SCHEDULER_TYPE == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config.EPOCHS, eta_min=self.config.LR_MIN
                )
            elif self.config.LR_SCHEDULER_TYPE == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        return optimizer, scheduler

    def _train_step(self, batch: tuple) -> float:
        """Performs one step of gradient descent."""
        images, _ = batch
        images = images.to(self.device)

        # Sample time steps t
        t = torch.randint(0, self.config.TIMESTEPS, (images.shape[0],), device=self.device, dtype=torch.long)

        # Generate noise
        noise = torch.randn_like(images)

        # Add noise to images (Forward Process)
        noisy_images = self.diffusion.q_sample(x_start=images, t=t, noise=noise)

        # Predict noise (Reverse Process)
        predicted_noise = self.model(noisy_images, t)

        # Calculate Loss (Simple MSE)
        loss = F.mse_loss(predicted_noise, noise)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
        self.optimizer.step()

        # Update EMA weights
        if self.ema:
            self.ema.update(self.model)

        return loss.item()

    def train(self):
        """Executes the main training loop."""
        print("🎉 Starting training! Let's create some art... 🎉")
        best_loss = float("inf")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.config.EPOCHS}", leave=False)
            for batch in pbar:
                loss = self._train_step(batch)
                epoch_loss += loss
                self.global_step += 1
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

            avg_loss = epoch_loss / len(self.dataloader)

            # End of epoch routines
            self._on_epoch_end(epoch, avg_loss, epoch_start_time, best_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

        self.logger.close()
        print("\n" + "---" * 10 + " ✨ TRAINING COMPLETE ✨ " + "---" * 10)
        print(f"  Best Loss: {best_loss:.4f}")
        print(f"  All logs and models saved in: '{self.run_dir}'")
        print("---" * 27 + "\n")

    def _on_epoch_end(self, epoch: int, avg_loss: float, epoch_start_time: float, best_loss: float):
        """Handles logging, saving, and sampling at the end of an epoch."""
        epoch_time_s = time.time() - epoch_start_time
        current_lr = self.optimizer.param_groups[0]["lr"]
        samples_per_sec = len(self.dataloader.dataset) / epoch_time_s

        # 1. Log metrics
        self.logger.log_epoch(
            {
                "epoch": epoch + 1,
                "step": self.global_step,
                "avg_loss": avg_loss,
                "learning_rate": current_lr,
                "epoch_time_s": epoch_time_s,
                "samples_per_sec": samples_per_sec,
            }
        )

        # 2. Step Scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        # 3. Save Checkpoint
        if (epoch + 1) % self.config.CHECKPOINT_EVERY == 0 or (epoch + 1) == self.config.EPOCHS:
            is_best = avg_loss < best_loss
            checkpoint_dict = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_state_dict": self.ema.state_dict() if self.ema else None,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            }
            checkpoint_name = f"epoch_{epoch + 1:03d}"
            self.logger.save_checkpoint(checkpoint_dict, checkpoint_name, is_best)

            status = "New best loss!" if is_best else "Saved."
            print(f"  💾 Checkpoint: {status}")

        # 4. Generate Samples
        if self.config.SAVE_SAMPLE_IMAGES and (epoch + 1) % self.config.SAMPLE_EVERY == 0:
            print("  🎨 Generating intermediate samples...")
            self._generate_samples()

        print(
            f"Epoch {epoch + 1:03d}/{self.config.EPOCHS} | Avg Loss: {avg_loss:.4f}"
            f" | LR: {current_lr:.2e} | Time: {epoch_time_s:.2f}s"
        )

    @torch.no_grad()
    def _generate_samples(self):
        """Generates visualization samples during training using the EMA model."""
        model_to_use = self.ema.model if self.ema else self.model
        model_to_use.eval()

        samples = self.diffusion.sample(
            model=model_to_use,
            shape=(
                self.config.NUM_SAMPLE_IMAGES,
                self.config.IMAGE_CHANNELS,
                self.config.IMAGE_SIZE,
                self.config.IMAGE_SIZE,
            ),
            method=self.config.SAMPLING_METHOD,
            sampling_steps=self.config.DDIM_SAMPLING_STEPS,
            eta=self.config.DDIM_ETA,
        )

        # Denormalize and Save
        samples = (samples.clamp(-1, 1) + 1) / 2
        save_path = self.logger.sample_dir / f"sample_epoch_{self.current_epoch + 1:03d}.png"
        save_image(samples, save_path, nrow=int(self.config.NUM_SAMPLE_IMAGES**0.5))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a Diffusion Model on MNIST.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, dest="learning_rate", default=None, help="Override learning rate.")

    args = parser.parse_args()

    config = Config()
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate

    config.print_config()
    trainer = Trainer(config, experiment_name=args.name)
    trainer.train()


if __name__ == "__main__":
    main()
