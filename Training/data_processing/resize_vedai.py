import os
from PIL import Image

# ===== PATH =====
depth_folder = r"C:/Users/ADMIN/Guided SR in thermal images/data/Vehicules512/val/depth"

# Nếu muốn làm cả val thì chạy thêm lần nữa với path val/depth

threshold = 688  # >= 00000688
target_size = (256, 256)

files = os.listdir(depth_folder)

count = 0

for file in files:
    if file.endswith("_ir.png"):
        try:
            index = int(file.split("_")[0])
        except:
            continue

        if index >= threshold:
            img_path = os.path.join(depth_folder, file)

            img = Image.open(img_path)

            # Resize
            img_resized = img.resize(target_size, Image.BICUBIC)

            # Ghi đè lại file cũ
            img_resized.save(img_path)

            print("Resized:", file)
            count += 1

print("Done. Total resized:", count)