import os
import shutil
import random

# ====== PATH ======
base_dir = r"C:/Users/ADMIN/Guided SR in thermal images/data/Vehicules512"

train_depth = os.path.join(base_dir, "train/depth")
train_gt = os.path.join(base_dir, "train/gt")
train_rgb = os.path.join(base_dir, "train/rgb")

val_depth = os.path.join(base_dir, "val/depth")
val_gt = os.path.join(base_dir, "val/gt")
val_rgb = os.path.join(base_dir, "val/rgb")


# ====== TÌM INDEX LỚN NHẤT ======
def get_max_index(folder):
    files = os.listdir(folder)
    indices = []
    for f in files:
        if f.endswith("_ir.png"):
            try:
                indices.append(int(f.split("_")[0]))
            except:
                pass
    return max(indices) if indices else -1


max_index = max(
    get_max_index(train_depth),
    get_max_index(val_depth)
)

print("Max index hiện tại:", max_index)

current_index = max_index + 1


# ====== LẤY ẢNH Ở ROOT ======
root_files = os.listdir(base_dir)

ir_files = [
    f for f in root_files
    if f.endswith("_ir.png") and os.path.isfile(os.path.join(base_dir, f))
]

print("Tìm thấy", len(ir_files), "ảnh mới.")

# Shuffle để chia 80/20
random.shuffle(ir_files)

split_index = int(len(ir_files) * 0.8)
train_files = ir_files[:split_index]
val_files = ir_files[split_index:]

print("Train mới:", len(train_files))
print("Val mới:", len(val_files))


# ====== HÀM MOVE ======
def process_files(file_list, depth_folder, gt_folder, rgb_folder, start_index):
    index = start_index

    for file in file_list:
        base_name = file.split("_")[0]

        ir_path = os.path.join(base_dir, f"{base_name}_ir.png")
        co_path = os.path.join(base_dir, f"{base_name}_co.png")

        if not os.path.exists(co_path):
            print("Thiếu file rgb:", base_name)
            continue

        new_name_ir = f"{index:08d}_ir.png"
        new_name_co = f"{index:08d}_co.png"

        # move depth
        shutil.move(ir_path, os.path.join(depth_folder, new_name_ir))

        # copy sang gt
        shutil.copy(
            os.path.join(depth_folder, new_name_ir),
            os.path.join(gt_folder, new_name_ir)
        )

        # move rgb
        shutil.move(co_path, os.path.join(rgb_folder, new_name_co))

        print("Added:", new_name_ir)

        index += 1

    return index


# ====== XỬ LÝ ======
current_index = process_files(
    train_files,
    train_depth,
    train_gt,
    train_rgb,
    current_index
)

current_index = process_files(
    val_files,
    val_depth,
    val_gt,
    val_rgb,
    current_index
)

print("Done.")