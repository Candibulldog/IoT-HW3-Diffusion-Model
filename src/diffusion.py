# src/diffusion.py

"""Diffusion Process Implementation.

This module implements the core mathematics of the Denoising Diffusion Probabilistic Models (DDPM).
It handles:
1. Beta schedules (Linear, Cosine).
2. Forward diffusion process (q_sample): Adding noise to images.
3. Reverse diffusion process (p_sample): Removing noise using the model.
4. Sampling strategies: DDPM (Ancestral Sampling) and DDIM (Deterministic/Accelerated).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extracts coefficients at specified time steps and reshapes for broadcasting.

    Args:
        a (torch.Tensor): Tensor containing the coefficients (e.g., sqrt_alphas).
        t (torch.Tensor): Tensor containing the time steps for each batch item.
        x_shape (tuple): Shape of the target tensor (Batch, Channels, Height, Width).

    Returns:
        torch.Tensor: Reshaped tensor compatible with x_shape (B, 1, 1, 1).
    """
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionScheduler:
    """Manages the diffusion noise schedules and sampling algorithms.

    Supports both 'linear' and 'cosine' beta schedules.
    Implements 'q_sample' for training and 'p_sample_ddpm'/'p_sample_ddim' for generation.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        device: str | torch.device = "cpu",
    ):
        """Initializes the DiffusionScheduler.

        Args:
            timesteps (int): Total number of diffusion steps.
            beta_schedule (str): 'linear' or 'cosine'.
            beta_start (float): Start value for linear beta schedule.
            beta_end (float): End value for linear beta schedule.
            cosine_s (float): Offset for cosine schedule.
            device (str | torch.device): Device to store precomputed constants.
        """
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == "linear":
            self.betas = self._linear_beta_schedule(beta_start, beta_end)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(cosine_s)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # --- Precompute Diffusion Constants ---
        # Alphas = 1 - Betas
        self.alphas = 1.0 - self.betas
        # Cumulative product of alphas (alpha_bar)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance calculation
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # Log calculation requires clamping to avoid -inf at t=0
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

        self._move_tensors_to_device()

    def _move_tensors_to_device(self):
        """Moves all registered tensor attributes to the configured device."""
        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(self.device))

    def _linear_beta_schedule(self, beta_start: float, beta_end: float) -> torch.Tensor:
        """Generates a linear sequence of betas."""
        return torch.linspace(beta_start, beta_end, self.timesteps)

    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """Generates a cosine schedule for betas.

        Proposed in "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021).
        Prevents abrupt noise changes at the end of the schedule.
        """
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Diffuse the data (add noise) -> Forward Process.

        Computes x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample_ddpm(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """Performs a single reverse diffusion step using DDPM (Ancestral Sampling).

        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta) + sigma_t * z
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Predict noise using the model
        predicted_noise = model(x, t)

        # Estimate mean of x_{t-1}
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_ddim(
        self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, eta: float
    ) -> torch.Tensor:
        """Performs a single reverse diffusion step using DDIM.

        Allows for non-Markovian sampling, enabling faster generation by skipping steps.
        Supports deterministic sampling when eta=0.
        """
        predicted_noise = model(x, t)

        alpha_t = extract(self.alphas_cumprod, t, x.shape)
        alpha_t_prev = (
            extract(self.alphas_cumprod, t_prev, x.shape) if (t_prev >= 0).all() else torch.ones_like(alpha_t)
        )

        # Predict x_0 based on current x_t and predicted noise
        pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Calculate direction pointing to x_t
        sigma_t = eta * torch.sqrt((1.0 - alpha_t_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_t_prev))
        dir_xt = torch.sqrt(1.0 - alpha_t_prev - sigma_t**2) * predicted_noise

        # Add noise if eta > 0
        noise = torch.randn_like(x) if eta > 0 and (t_prev > 0).any() else 0

        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        method: str = "ddim",
        sampling_steps: int = 50,
        eta: float = 0.0,
        return_all_steps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generates images from pure noise using the specified sampling method.

        Args:
            model (nn.Module): The trained U-Net model.
            shape (tuple): Desired output shape (B, C, H, W).
            method (str): 'ddpm' or 'ddim'.
            sampling_steps (int): Number of steps for DDIM.
            eta (float): Controls stochasticity in DDIM (0.0 = deterministic).
            return_all_steps (bool): If True, returns a list of intermediate images.

        Returns:
            torch.Tensor: Final generated images.
            (optional) list[torch.Tensor]: History of images if return_all_steps is True.
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        images_over_time = [x.cpu()] if return_all_steps else None

        # Determine schedule
        if method == "ddpm":
            timesteps_seq = reversed(range(self.timesteps))
        elif method == "ddim":
            # Linearly space the timesteps
            timesteps_seq = torch.linspace(self.timesteps - 1, 0, steps=sampling_steps, dtype=torch.long).tolist()
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        for i, t_val in enumerate(tqdm(timesteps_seq, desc=f"{method.upper()} Sampling", leave=False)):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            if method == "ddpm":
                x = self.p_sample_ddpm(model, x, t, t_val)
            elif method == "ddim":
                t_prev_val = timesteps_seq[i + 1] if i + 1 < len(timesteps_seq) else -1
                t_prev = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)
                x = self.p_sample_ddim(model, x, t, t_prev, eta)

            if return_all_steps:
                images_over_time.append(x.cpu())

        if return_all_steps:
            return x, images_over_time
        return x

    @torch.no_grad()
    def sample_for_demo(
        self,
        model: nn.Module,
        shape: tuple,
        capture_every: int = 50,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generates images and captures intermediate states for visualization.

        This method is specifically designed for Streamlit demonstrations to
        show the gradual denoising process. It defaults to DDPM sampling for
        smooth trajectory visualization.

        Args:
            model (nn.Module): The trained U-Net model.
            shape (tuple): Desired output shape (Batch, Channels, Height, Width).
            capture_every (int): Interval of timesteps to capture a snapshot.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                - The final generated image tensor.
                - A list of intermediate image tensors captured during the process.
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        # Initialize history with pure noise
        intermediates = [x.cpu().clone()]

        # Iterate backwards from T-1 to 0 (DDPM Standard)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # Perform one DDPM step
            x = self.p_sample_ddpm(model, x, t, i)

            # Capture intermediate state
            # Capture if it matches the interval or if it's the final clean step (i=0)
            if i % capture_every == 0 or i == 0:
                intermediates.append(x.cpu().clone())

        return x, intermediates
