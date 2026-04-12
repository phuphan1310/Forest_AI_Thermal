import os
import shutil

base_dir = r"C:/Users/ADMIN/MySGNet/data/kaist-cvpr15"
images_dir = os.path.join(base_dir, "images")

out_lwir = os.path.join(base_dir, "lwir")
out_visible = os.path.join(base_dir, "visible")

os.makedirs(out_lwir, exist_ok=True)
os.makedirs(out_visible, exist_ok=True)

total_count = 0
missing_count = 0

for set_name in os.listdir(images_dir):
    set_path = os.path.join(images_dir, set_name)
    if not os.path.isdir(set_path):
        continue

    for v_name in os.listdir(set_path):
        v_path = os.path.join(set_path, v_name)
        if not os.path.isdir(v_path):
            continue

        lwir_path = os.path.join(v_path, "lwir")
        visible_path = os.path.join(v_path, "visible")

        if not os.path.exists(lwir_path) or not os.path.exists(visible_path):
            continue

        # Lấy danh sách file jpg thật
        lwir_files = {
            f for f in os.listdir(lwir_path)
            if f.lower().endswith(".jpg") and not f.startswith("._")
        }

        visible_files = {
            f for f in os.listdir(visible_path)
            if f.lower().endswith(".jpg") and not f.startswith("._")
        }

        # Chỉ lấy phần giao nhau (ảnh tồn tại ở cả 2 folder)
        common_files = lwir_files.intersection(visible_files)

        # Đếm số file bị thiếu
        missing = len(lwir_files.symmetric_difference(visible_files))
        missing_count += missing

        for filename in common_files:
            new_name = f"{set_name}_{v_name}_{filename}"

            shutil.copy(
                os.path.join(lwir_path, filename),
                os.path.join(out_lwir, new_name)
            )

            shutil.copy(
                os.path.join(visible_path, filename),
                os.path.join(out_visible, new_name)
            )

            total_count += 1

print("Done!")
print(f"Tổng số cặp ảnh hợp lệ: {total_count}")
print(f"Số ảnh bị thiếu (không đủ cặp): {missing_count}")
