# src/ema.py

"""
Exponential Moving Average (EMA) for model parameters.
EMA significantly improves generation quality by smoothing model weights.
"""

from typing import Any

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow model whose weights are a moving average of the trained
    model's weights, often leading to better and more stable generation results.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, update_every: int = 10):
        """
        Args:
            model: The model to track.
            decay: EMA decay rate (higher = slower update, more smoothing).
            update_every: Update EMA weights every N steps for efficiency.
        """
        # Create a new model instance of the same class on the correct device
        # This avoids the complex CPU-GPU transfers of deepcopy
        self.ema_model = type(model)(**model.config).to(next(model.parameters()).device)
        self.ema_model.load_state_dict(model.state_dict())

        # Configure the EMA model
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

        self.decay = decay
        self.update_every = update_every
        self.step = 0

        print(f"✅ EMA initialized on device: {next(self.ema_model.parameters()).device}")

    def update(self, model: nn.Module):
        """
        Updates the EMA model's parameters with the current model's parameters.
        """
        self.step += 1
        if self.step % self.update_every != 0:
            return

        with torch.no_grad():
            model_state_dict = model.state_dict()
            for name, ema_param in self.ema_model.state_dict().items():
                model_param = model_state_dict[name]
                # The device check is implicitly handled as both models are on the same device
                ema_param.mul_(self.decay).add_(model_param, alpha=1 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        """Returns the state dictionary of the EMA model."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads a state dictionary into the EMA model."""
        self.ema_model.load_state_dict(state_dict)

    @property
    def model(self) -> nn.Module:
        """Property to access the EMA model directly."""
        return self.ema_model

    def reset(self, model: nn.Module):
        """Resets the EMA model's weights to the current model's weights."""
        self.load_state_dict(model.state_dict())
        self.step = 0


if __name__ == "__main__":
    """Tests the EMA functionality."""
    # This test script remains the same as it tests the external behavior,
    # which has not changed.
    print("🔬 Testing EMA implementation...")

    # We need to add a config attribute to the model for the new init to work
    class SimpleModel(nn.Module):
        def __init__(self, in_features=2, out_features=2):
            super().__init__()
            self.config = {"in_features": in_features, "out_features": out_features}
            self.linear = nn.Linear(in_features, out_features)
            self.register_buffer("counter", torch.tensor(0.0))

    model = SimpleModel()

    print("\n[1/2] Verifying EMA update formula...")
    ema = EMA(model, decay=0.9, update_every=1)

    model.linear.weight.data.fill_(1.0)
    model.counter.data.fill_(1.0)
    ema.update(model)
    assert torch.allclose(ema.model.linear.weight.data, torch.tensor(0.1))
    assert torch.allclose(ema.model.counter.data, torch.tensor(0.1))
    print("✅ EMA update after first step is correct.")

    model.linear.weight.data.fill_(2.0)
    model.counter.data.fill_(2.0)
    ema.update(model)
    assert torch.allclose(ema.model.linear.weight.data, torch.tensor(0.29))
    assert torch.allclose(ema.model.counter.data, torch.tensor(0.29))
    print("✅ EMA update after second step is correct.")

    print("\n[2/2] Verifying general functionality...")
    ema.reset(model)
    assert torch.allclose(ema.model.linear.weight.data, torch.tensor(2.0))
    print("✅ 'reset' method works.")

    state = ema.state_dict()
    new_ema = EMA(SimpleModel())
    new_ema.load_state_dict(state)
    assert torch.allclose(new_ema.model.linear.weight.data, torch.tensor(2.0))
    print("✅ 'state_dict' and 'load_state_dict' work.")

    print("\n🎉 EMA class is working as expected!")
