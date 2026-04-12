import os
import shutil
import random

# ====== THIẾT LẬP ======
base_dir = r"C:/Users/ADMIN/MySGNet/data/sugarbeet2016/train"

rgb_dir = os.path.join(base_dir, "rgb")
gt_dir = os.path.join(base_dir, "gt")
depth_dir = os.path.join(base_dir, "depth")

val_rgb_dir = os.path.join(base_dir, "val", "rgb")
val_gt_dir = os.path.join(base_dir, "val", "gt")
val_depth_dir = os.path.join(base_dir, "val", "depth")

num_val = 200
# ========================

os.makedirs(val_rgb_dir, exist_ok=True)
os.makedirs(val_gt_dir, exist_ok=True)
os.makedirs(val_depth_dir, exist_ok=True)

gt_files = [f for f in os.listdir(gt_dir)
            if f.lower().endswith((".png", ".jpg", ".tif", ".tiff"))]

valid_triplets = []

for gt_file in gt_files:

    if "_nir_" not in gt_file:
        continue

    rgb_file = gt_file.replace("_nir_", "_rgb_")
    depth_file = gt_file  # depth giống gt

    if (os.path.exists(os.path.join(rgb_dir, rgb_file)) and
        os.path.exists(os.path.join(depth_dir, depth_file))):

        valid_triplets.append((gt_file, rgb_file, depth_file))

print(f"Tổng bộ hợp lệ: {len(valid_triplets)}")

if len(valid_triplets) < num_val:
    print("❌ Không đủ 200 bộ ảnh hợp lệ.")
    exit()

random.seed(42)
selected = random.sample(valid_triplets, num_val)

for gt_file, rgb_file, depth_file in selected:

    shutil.move(os.path.join(gt_dir, gt_file),
                os.path.join(val_gt_dir, gt_file))

    shutil.move(os.path.join(rgb_dir, rgb_file),
                os.path.join(val_rgb_dir, rgb_file))

    shutil.move(os.path.join(depth_dir, depth_file),
                os.path.join(val_depth_dir, depth_file))

    print(f"✔ Moved: {gt_file}")

print("=== Hoàn tất tạo val set ===")
