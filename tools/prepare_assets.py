# tools/prepare_assets.py
# 在你的 RTX 4070 電腦上執行這個
import torch
import sys
from pathlib import Path

sys.path.append(".")  # 確保能讀到 src

from src.config import Config
from src.generate import ImageGenerator
from torchvision.utils import save_image


def main():
    # 1. 載入模型
    model_path = Path("checkpoints/model.pth")  # 你的模型路徑
    config = Config()
    config.DEVICE = "cuda"
    generator = ImageGenerator(model_path, config, use_ema=True)

    # 2. 設定要預先生成的 Seed
    seeds = [42, 123, 777, 2024, 999]  # 選 5 個漂亮的

    output_root = Path("assets/demo_cache")
    output_root.mkdir(parents=True, exist_ok=True)

    print("開始預計算 Demo 素材...")

    for seed in seeds:
        print(f"Processing Seed {seed}...")
        torch.manual_seed(seed)

        # 準備資料夾: assets/demo_cache/seed_42/
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        shape = (1, 1, 28, 28)  # MNIST

        # 使用我們之前寫的 sample_for_demo，但這次間隔設小一點，例如 20，讓動畫更順
        # 注意：你需要確保 diffusion.py 的 sample_for_demo 有被正確 import
        final_img, history = generator.diffusion.sample_for_demo(
            generator.model, shape, capture_every=20
        )

        # 儲存每一張圖
        # history[0] 是雜訊 (T), history[-1] 是結果 (0)
        total_steps = len(history)
        for i, img_tensor in enumerate(history):
            # 檔名存成 step_00.png, step_01.png ... 方便排序
            # 對應的 timestep
            t = config.TIMESTEPS - (i * 20)
            if i == len(history) - 1:
                t = 0

            save_image(img_tensor, seed_dir / f"step_{i:03d}_t{t}.png")

    print("素材準備完成！")


if __name__ == "__main__":
    main()
