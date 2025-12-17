# src/experiment_logger.py

"""Experiment Logger module.

This module provides a lightweight, file-based logging system for tracking
diffusion model training. It handles:
1. Creating experiment directories.
2. Saving hyperparameters (config.json).
3. Logging training metrics to CSV.
4. Managing model checkpoint storage.
"""

import csv
import json
from pathlib import Path
from typing import Any

import torch


class ExperimentLogger:
    """Handles logging for a single experiment run.

    Creates a structured directory layout:
    /log_dir
        /checkpoints
        /samples
        config.json
        training_log.csv
    """

    def __init__(self, log_dir: Path):
        """Initializes the logger and creates necessary directories.

        Args:
            log_dir (Path): The root directory for this specific experiment run.
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.sample_dir = self.log_dir / "samples"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.sample_dir.mkdir(exist_ok=True)

        self.csv_path = self.log_dir / "training_log.csv"
        self._init_csv()

        print("\n--- 📝 LOGGER INITIALIZED 📝 ---")
        print(f"  Logging experiment to: '{self.log_dir}'")
        print("---------------------------------\n")

    def _init_csv(self):
        """Initializes the CSV log file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "step", "avg_loss", "learning_rate", "epoch_time_s", "samples_per_sec"])

    def save_config(self, config_obj: Any):
        """Saves the experiment configuration to a JSON file.

        Handles serialization of non-JSON types like Path, torch.device, and tuples.

        Args:
            config_obj (Any): The Config class or instance to save.
        """
        serializable_config = {}

        # Determine if config_obj is an instance or a class
        config_class = type(config_obj) if not isinstance(config_obj, type) else config_obj

        # Iterate through attributes defined in the Config
        for key in dir(config_class):
            # Skip private attributes and magic methods
            if key.startswith("_"):
                continue

            # Get value (handles both instance overrides and class defaults)
            value = getattr(config_obj, key)

            # Skip methods
            if callable(value):
                continue

            # Convert non-serializable types
            if isinstance(value, Path):
                serializable_config[key] = str(value)
            elif isinstance(value, torch.device):
                serializable_config[key] = str(value)
            elif isinstance(value, (tuple, set)):
                serializable_config[key] = list(value)
            else:
                serializable_config[key] = value

        config_path = self.log_dir / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(serializable_config, f, indent=4, sort_keys=True)
            print(f"✅ Configuration successfully saved to: {config_path}")
        except Exception as e:
            print(f"❌ Error saving config to JSON: {e}")

    def log_epoch(self, epoch_data: dict[str, Any]) -> None:
        """Logs metrics for a completed epoch to the CSV file.

        Args:
            epoch_data (dict): Dictionary containing 'epoch', 'step', 'avg_loss', etc.
        """
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch_data["epoch"],
                    epoch_data["step"],
                    f"{epoch_data['avg_loss']:.6f}",
                    f"{epoch_data['learning_rate']:.8f}",
                    f"{epoch_data['epoch_time_s']:.2f}",
                    f"{epoch_data.get('samples_per_sec', 0):.1f}",
                ]
            )

    def save_checkpoint(self, checkpoint_dict: dict[str, Any], name: str, is_best: bool = False):
        """Saves a model checkpoint to disk.

        Args:
            checkpoint_dict (dict): The dictionary containing model state, optimizer state, etc.
            name (str): The identifier for the checkpoint (e.g., 'epoch_10').
            is_best (bool, optional): If True, saves an additional copy as 'best_checkpoint.pth'.
        """
        filename = f"checkpoint_{name}.pth"
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint_dict, save_path)

        if is_best:
            torch.save(checkpoint_dict, self.checkpoint_dir / "best_checkpoint.pth")

    def close(self) -> None:
        """Finalizes logging (placeholder for potential resource cleanup)."""
        print(f"\n✅ Experiment logs saved successfully in '{self.log_dir}'")


if __name__ == "__main__":
    """Run a quick test of the logger functionality."""
    import shutil

    test_root = Path("./temp_logger_root")
    exp_dir = test_root / "test_experiment"

    try:
        print("🔬 Testing ExperimentLogger...")
        logger = ExperimentLogger(log_dir=exp_dir)

        # Test config saving
        logger.save_config(type("MockConfig", (), {"LEARNING_RATE": 1e-3, "PATH": Path("./test")})())

        # Test logging
        for i in range(3):
            logger.log_epoch(
                {
                    "epoch": i + 1,
                    "step": (i + 1) * 10,
                    "avg_loss": 0.5 / (i + 1),
                    "learning_rate": 1e-3,
                    "epoch_time_s": 1.5,
                }
            )

        logger.close()
        print("✅ Logger test passed.")

    except Exception as e:
        print(f"❌ Logger test failed: {e}")
    finally:
        if test_root.exists():
            shutil.rmtree(test_root)
            print("🧹 Cleaned up test directory.")
