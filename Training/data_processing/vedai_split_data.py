import os
import shutil

base_dir = r"C:/Users/ADMIN/MySGNet/data/Vehicules512"

rgb_dir = os.path.join(base_dir, "rgb")
gt_dir = os.path.join(base_dir, "gt")

# Tạo folder nếu chưa có
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

files = os.listdir(base_dir)

co_count = 0
ir_count = 0

for file in files:
    if not file.lower().endswith(".png"):
        continue

    src_path = os.path.join(base_dir, file)

    if file.endswith("_co.png"):
        shutil.move(src_path, os.path.join(rgb_dir, file))
        co_count += 1

    elif file.endswith("_ir.png"):
        shutil.move(src_path, os.path.join(gt_dir, file))
        ir_count += 1

print("Hoàn tất!")
print(f"Số ảnh rgb (_co): {co_count}")
print(f"Số ảnh gt (_ir): {ir_count}")
