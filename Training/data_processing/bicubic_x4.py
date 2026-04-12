import cv2
import os

# ====== THIẾT LẬP ======
input_dir = r"C:/Users/ADMIN/SGNet/data/FLIR_ADAS_1_3/val/thermal_8_bit_LR_x4"
output_dir = r"C:/Users/ADMIN/SGNet/data/FLIR_ADAS_1_3/val/thermal_16_bit_LR_x4_bicubic"
scale = 4   # phải khớp với scale downsample trước đó
# ========================

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpeg",".png", ".tiff", ".tif")):

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Đọc ảnh giữ nguyên 16-bit
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"❌ Không đọc được: {filename}")
            continue

        if img.dtype != 'uint16':
            print(f"⚠ Cảnh báo: {filename} không phải uint16")

        h, w = img.shape[:2]

        # Upsample về kích thước gốc
        up = cv2.resize(
            img,
            (w * scale, h * scale),
            interpolation=cv2.INTER_CUBIC
        )

        cv2.imwrite(output_path, up)

        print(f"✔ Done: {filename}")

print("=== Hoàn tất bicubic upsample ===")
