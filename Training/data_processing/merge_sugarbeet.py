import os
import shutil

base_dir = r"C:/Users/ADMIN/MySGNet/data/sugarbeet2016"
raw_dir = os.path.join(base_dir, "raw_data")

out_gt = os.path.join(base_dir, "gt")
out_rgb = os.path.join(base_dir, "rgb")

os.makedirs(out_gt, exist_ok=True)
os.makedirs(out_rgb, exist_ok=True)

total_nir = 0
total_rgb = 0

for folder in os.listdir(raw_dir):
    session_path = os.path.join(raw_dir, folder)

    if not os.path.isdir(session_path):
        continue

    jai_path = os.path.join(session_path, "camera", "jai")
    nir_path = os.path.join(jai_path, "nir")
    rgb_path = os.path.join(jai_path, "rgb")

    if not os.path.exists(nir_path) or not os.path.exists(rgb_path):
        continue

    # Gộp NIR -> gt
    for file in os.listdir(nir_path):
        if not file.lower().endswith((".png", ".jpg")):
            continue

        new_name = f"{folder}_{file}"
        shutil.copy(
            os.path.join(nir_path, file),
            os.path.join(out_gt, new_name)
        )
        total_nir += 1

    # Gộp RGB
    for file in os.listdir(rgb_path):
        if not file.lower().endswith((".png", ".jpg")):
            continue

        new_name = f"{folder}_{file}"
        shutil.copy(
            os.path.join(rgb_path, file),
            os.path.join(out_rgb, new_name)
        )
        total_rgb += 1

print("Hoàn tất!")
print(f"Tổng NIR (gt): {total_nir}")
print(f"Tổng RGB: {total_rgb}")
