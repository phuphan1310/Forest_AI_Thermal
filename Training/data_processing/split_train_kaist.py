import os
import random
import shutil

base_dir = r"C:/Users/ADMIN/MySGNet/data/kaist-cvpr15"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

train_gt = os.path.join(train_dir, "gt")
train_rgb = os.path.join(train_dir, "rgb")
train_depth = os.path.join(train_dir, "depth")

val_gt = os.path.join(val_dir, "gt")
val_rgb = os.path.join(val_dir, "rgb")
val_depth = os.path.join(val_dir, "depth")

# Tạo folder val nếu chưa có
os.makedirs(val_gt, exist_ok=True)
os.makedirs(val_rgb, exist_ok=True)
os.makedirs(val_depth, exist_ok=True)

# Lấy danh sách ảnh từ gt (vì 3 folder đã đồng bộ)
all_images = [f for f in os.listdir(train_gt) if f.lower().endswith(".jpg")]

if len(all_images) < 120:
    raise ValueError("Không đủ 120 ảnh để chia!")

# Chọn ngẫu nhiên 120 ảnh
selected = random.sample(all_images, 120)

print(f"Chuyển {len(selected)} ảnh sang val...")

for img_name in selected:
    # Di chuyển cả 3 folder
    shutil.move(os.path.join(train_gt, img_name),
                os.path.join(val_gt, img_name))

    shutil.move(os.path.join(train_rgb, img_name),
                os.path.join(val_rgb, img_name))

    shutil.move(os.path.join(train_depth, img_name),
                os.path.join(val_depth, img_name))

print("Hoàn tất!")
