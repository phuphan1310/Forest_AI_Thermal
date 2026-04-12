import os
import random
from PIL import Image

base_dir = r"C:/Users/ADMIN/MySGNet/data/kaist-cvpr15"

gt_dir = os.path.join(base_dir, "gt")
rgb_dir = os.path.join(base_dir, "rgb")
depth_dir = os.path.join(base_dir, "depth")

os.makedirs(depth_dir, exist_ok=True)

# Lấy toàn bộ ảnh jpg
all_images = [f for f in os.listdir(gt_dir) if f.lower().endswith(".jpg")]

# Gom ảnh theo từng set
sets = {}
for img in all_images:
    set_name = img.split("_")[0]  # ví dụ set00
    sets.setdefault(set_name, []).append(img)

print("Bắt đầu xử lý...")

for set_name in sorted(sets.keys()):
    images = sets[set_name]

    if len(images) < 100:
        print(f"{set_name} không đủ 100 ảnh, bỏ qua.")
        continue

    # Chọn ngẫu nhiên 100 ảnh
    selected = random.sample(images, 100)

    print(f"{set_name}: chọn {len(selected)} ảnh")

    for img_name in selected:
        gt_path = os.path.join(gt_dir, img_name)

        # Resize và lưu vào depth
        img = Image.open(gt_path)
        img_resized = img.resize((160, 128), Image.BILINEAR)

        depth_path = os.path.join(depth_dir, img_name)
        img_resized.save(depth_path)

    # Xóa các ảnh không được chọn trong gt và rgb
    for img_name in images:
        if img_name not in selected:
            gt_path = os.path.join(gt_dir, img_name)
            rgb_path = os.path.join(rgb_dir, img_name)

            if os.path.exists(gt_path):
                os.remove(gt_path)
            if os.path.exists(rgb_path):
                os.remove(rgb_path)

print("Hoàn tất!")
