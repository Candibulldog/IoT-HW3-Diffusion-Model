# app.py

"""Streamlit Application for Diffusion Model Demonstration.

This application serves as the interactive demo for the Deep Learning homework.
It visualizes the reverse diffusion process (denoising) of a trained model on MNIST.

Run command:
    streamlit run app.py
"""

import sys
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import torch

# Ensure the local src module is accessible
sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.generate import ImageGenerator

# --- Page Configuration ---
st.set_page_config(
    page_title="Diffusion Model Demo",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Utility Functions ---
def normalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts a [-1, 1] tensor to a [0, 255] numpy array for display.

    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W).

    Returns:
        np.ndarray: Image array of shape (H, W, C) or (H, W) suitable for PIL/Streamlit.
    """
    # Clamp and rescale from [-1, 1] to [0, 1]
    img = (tensor.clamp(-1, 1) + 1) / 2
    # Convert to CPU numpy
    img = img.cpu().permute(1, 2, 0).numpy()
    # Scale to [0, 255]
    img = (img * 255).astype(np.uint8)

    # Squeeze channel dimension if grayscale (H, W, 1) -> (H, W)
    if img.shape[2] == 1:
        img = img.squeeze(2)

    return img


@st.cache_resource
def download_model_if_missing(model_path: Path):
    """Checks if model exists, if not, downloads from Google Drive."""
    if model_path.exists():
        return

    # Create the directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)

    st.warning(f"Model not found at {model_path}. Downloading from Google Drive (approx 600MB)...")

    # ==========================================
    # ⚠️ 請在下方填入你的 Google Drive File ID ⚠️
    # ==========================================
    # 例如連結是 https://drive.google.com/file/d/1A2B3C.../view
    # ID 就是 '1A2B3C...'
    GOOGLE_DRIVE_FILE_ID = "149W929n24DoCwzUWY_trlwG5lcpom4cF"

    if GOOGLE_DRIVE_FILE_ID == "149W929n24DoCwzUWY_trlwG5lcpom4cF":
        st.error("Error: Please enter the correct Google Drive File ID in app.py!")
        st.stop()

    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

    try:
        # download the file
        gdown.download(url, str(model_path), quiet=False)
        st.success("Download complete!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()


@st.cache_resource
def load_generator(model_path_str: str, device_str: str) -> ImageGenerator:
    """Loads the ImageGenerator and caches it to avoid reloading on every interaction.

    Args:
        model_path_str (str): Path to the model checkpoint.
        device_str (str): 'cpu' or 'cuda'.

    Returns:
        ImageGenerator: Initialized generator with loaded weights.
    """
    # Create a config override for the demo
    config = Config()
    config.DEVICE = device_str
    config.GENERATION_BATCH_SIZE = 1  # We only generate one image at a time for the demo

    model_path = Path(model_path_str)

    # --- 新增：自動下載邏輯 ---
    # 如果是預設路徑，嘗試自動下載
    if "checkpoints/model.pth" in str(model_path):
        download_model_if_missing(model_path)

    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    # Initialize generator (Use EMA if available for better quality)
    generator = ImageGenerator(model_path=model_path, config=config, use_ema=True)
    return generator


# --- Main Layout ---


def main():
    # --- Sidebar: Configuration ---
    st.sidebar.title("⚙️ Parameters")

    # Path to your checkpoint (Make sure this path is correct relative to app.py)
    # You can change the default value here
    default_model_path = "checkpoints/model.pth"
    model_path_input = st.sidebar.text_input("Model Checkpoint Path", value=default_model_path)

    # Device selection (Auto-detect usually works, but allow manual override)
    device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device_selection = st.sidebar.selectbox("Compute Device", device_options)

    # Generation Parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Generation Settings")
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    # Visualization Parameters
    st.sidebar.subheader("Visualization")
    capture_interval = st.sidebar.slider(
        "Snapshot Interval",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="How often (in timesteps) to capture an intermediate image.",
    )

    # --- Load Model ---
    try:
        with st.spinner(f"Loading model from {model_path_input}..."):
            generator = load_generator(model_path_input, device_selection)
        st.sidebar.success("Model Loaded Successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()

    # --- Main Content ---
    st.title("🖌️ Deep Learning Project: Diffusion Model")

    # 1. ABSTRACT SECTION (Requirement 1)
    with st.expander("📝 Project Abstract (Click to Expand)", expanded=True):
        st.markdown(
            """
            **Abstract:**

            This project implements a Denoising Diffusion Probabilistic Model (DDPM) trained on the MNIST dataset.
            The goal is to generate high-fidelity handwritten digits by reversing a gradual noise addition process.

            The model architecture is based on a U-Net with attention mechanisms at lower resolutions.
            The training objective maximizes the evidence lower bound (ELBO) by predicting the noise added to the image at each timestep $t$.

            This demonstration extends the standard training by visualizing the Langevin dynamics/reverse diffusion process,
            allowing users to observe how structured data emerges from pure Gaussian noise.

            *(Paste your actual 300-word abstract here for the homework)*
            """
        )

    st.divider()

    # 2. GENERATION SECTION
    st.header("1. Image Generation & Trajectory")

    col_btn, col_info = st.columns([1, 4])

    with col_btn:
        generate_btn = st.button("Generate Digit", type="primary", use_container_width=True)

    with col_info:
        st.info(f"Current Settings: Seed={seed}, Interval={capture_interval}")

    if generate_btn:
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Prepare shape (Batch=1, C, H, W)
        shape = (1, generator.config.IMAGE_CHANNELS, generator.config.IMAGE_SIZE, generator.config.IMAGE_SIZE)

        # Run Sampling
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Running Reverse Diffusion Process...")

        # Call the new sample_for_demo method
        try:
            final_tensor, history = generator.diffusion.sample_for_demo(
                model=generator.model, shape=shape, capture_every=capture_interval
            )
            progress_bar.progress(100)
            status_text.text("Generation Complete!")

        except AttributeError:
            st.error(
                "Error: `sample_for_demo` method not found in DiffusionScheduler. Please update `src/diffusion.py`."
            )
            st.stop()

        # Display Result
        st.subheader("Result")

        # Layout: Left for Final Image, Right for Process Strip
        col_final, col_strip = st.columns([1, 3])

        with col_final:
            st.markdown("**Final Output ($x_0$)**")
            final_img = normalize_image(final_tensor)
            st.image(final_img, width=150, clamp=True)

        with col_strip:
            st.markdown("**Denoising Trajectory ($x_T \\rightarrow x_0$)**")

            # Display history images side-by-side
            # The history list is [x_T, ..., x_0].
            # We want to show them from Noise -> Clean.

            # Create dynamic columns based on number of snapshots
            cols = st.columns(len(history))

            for idx, step_tensor in enumerate(history):
                # Calculate the approximate timestep for this snapshot
                # history[0] is pure noise (T), history[-1] is clean (0)
                if idx == 0:
                    caption = "T (Noise)"
                elif idx == len(history) - 1:
                    caption = "0 (Clean)"
                else:
                    # Estimate T based on index
                    # This is approximate just for display
                    t_est = generator.config.TIMESTEPS - (idx * capture_interval)
                    caption = f"t≈{t_est}"

                img_np = normalize_image(step_tensor)
                with cols[idx]:
                    st.image(img_np, caption=caption, use_container_width=True)

    st.divider()

    # 3. TECHNICAL DETAILS SECTION (Optional but good for reports)
    st.header("2. Model & Agent Details")
    st.json(
        {
            "Model Architecture": "U-Net",
            "Timesteps": generator.config.TIMESTEPS,
            "Image Size": f"{generator.config.IMAGE_SIZE}x{generator.config.IMAGE_SIZE}",
            "Sampling Method": "DDPM (Ancestral Sampling)",
            "Device": str(generator.device),
        }
    )


if __name__ == "__main__":
    main()
