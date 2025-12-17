# app.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Diffusion Demo", layout="wide")

st.title("🖌️ MNIST Diffusion Model (Interactive Demo)")
st.markdown("""
**Abstract:** This demo visualizes the reverse diffusion process where structured data (digits) 
emerges from pure Gaussian noise. The trajectory is pre-computed to ensure smooth interaction.
""")

# 1. 讀取素材資料夾
ASSETS_DIR = Path("assets/demo_cache")

if not ASSETS_DIR.exists():
    st.error(
        "⚠️ 找不到素材資料夾 `assets/demo_cache`。請先執行 `tools/prepare_assets.py`。"
    )
    st.stop()

# 取得所有可用的 Seed (根據資料夾名稱)
# 資料夾命名格式需為 seed_XXX
available_seeds = []
for d in ASSETS_DIR.iterdir():
    if d.is_dir() and d.name.startswith("seed_"):
        try:
            seed_val = int(d.name.split("_")[1])
            available_seeds.append(seed_val)
        except ValueError:
            continue

available_seeds = sorted(available_seeds)

if not available_seeds:
    st.error("⚠️ 資料夾內沒有有效的 Seed 資料。")
    st.stop()

# 2. 側邊欄：選擇種子
st.sidebar.header("控制參數")
selected_seed = st.sidebar.selectbox("選擇種子 (Select Seed)", available_seeds)

# 讀取該 Seed 的所有圖片
seed_dir = ASSETS_DIR / f"seed_{selected_seed}"
image_files = sorted(list(seed_dir.glob("*.png")))  # 確保按 step_000, step_001 排序

if not image_files:
    st.error(f"Seed {selected_seed} 資料夾是空的。")
    st.stop()

# 3. 主畫面展示
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("最終結果 (Final Result)")
    # 顯示最後一張圖 (去噪完成)
    final_img_path = image_files[-1]
    # 使用 use_container_width (舊版是 use_column_width)
    st.image(str(final_img_path), width=300, caption=f"Seed: {selected_seed} (Clean)")

with col2:
    st.subheader("擴散過程 (Denoising Trajectory)")

    # 建立滑桿
    # 範圍從 0 到 len-1
    step_idx = st.slider(
        "拖動滑桿觀察雜訊消除過程",
        min_value=0,
        max_value=len(image_files) - 1,
        value=len(image_files) - 1,
    )

    current_img_path = image_files[step_idx]

    # 解析檔名取得 t 值 (假設檔名 step_005_t800.png)
    # 這樣顯示起來更專業
    try:
        t_val = current_img_path.stem.split("_t")[1]
        caption = f"Timestep t = {t_val}"
    except:
        caption = f"Step {step_idx}"

    st.image(str(current_img_path), width=300, caption=caption)

st.info("💡 Note: This is a pre-computed demonstration running without GPU inference.")
