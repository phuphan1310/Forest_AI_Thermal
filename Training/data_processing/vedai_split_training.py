import os
import random
import shutil

base_dir = r"C:/Users/ADMIN/MySGNet/data/Vehicules512"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

train_rgb = os.path.join(train_dir, "rgb")
train_gt = os.path.join(train_dir, "gt")
train_depth = os.path.join(train_dir, "depth")

val_rgb = os.path.join(val_dir, "rgb")
val_gt = os.path.join(val_dir, "gt")
val_depth = os.path.join(val_dir, "depth")

os.makedirs(val_rgb, exist_ok=True)
os.makedirs(val_gt, exist_ok=True)
os.makedirs(val_depth, exist_ok=True)

# Hàm lấy ID số từ tên file
def get_id(filename):
    return filename.split("_")[0]

# Tạo dict: id -> filename
rgb_dict = {get_id(f): f for f in os.listdir(train_rgb) if f.endswith(".png")}
gt_dict = {get_id(f): f for f in os.listdir(train_gt) if f.endswith(".png")}
depth_dict = {get_id(f): f for f in os.listdir(train_depth) if f.endswith(".png")}

# Lấy ID chung
common_ids = list(set(rgb_dict.keys()) & set(gt_dict.keys()) & set(depth_dict.keys()))

if len(common_ids) < 100:
    raise ValueError("Không đủ 100 ID có đủ cả 3 loại ảnh!")

# Chọn 100 ID ngẫu nhiên
selected_ids = random.sample(common_ids, 100)

print(f"Chuyển {len(selected_ids)} ID sang val...")

for img_id in selected_ids:
    shutil.move(os.path.join(train_rgb, rgb_dict[img_id]),
                os.path.join(val_rgb, rgb_dict[img_id]))

    shutil.move(os.path.join(train_gt, gt_dict[img_id]),
                os.path.join(val_gt, gt_dict[img_id]))

    shutil.move(os.path.join(train_depth, depth_dict[img_id]),
                os.path.join(val_depth, depth_dict[img_id]))

print("Hoàn tất!")
