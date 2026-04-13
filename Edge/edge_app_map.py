import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import json
import time
import torch
import requests
import gc
import random
import glob
import re
from models.SGNet import SGNet

# ==================== CẤU HÌNH HỆ THỐNG ====================
SERVER_IP = "100.95.139.91"
REPORT_URL = f"http://{SERVER_IP}:5001/report"
TX_LAT = 21.0000
TX_LON = 105.8150
SIMULATE_GPS = True

# Thư mục dữ liệu (thay đổi theo đường dẫn thực tế của bạn)
RGB_DIR = "/home/dhcn/thermal_project/rgb"          # Ví dụ: thư mục chứa ảnh RGB
THERMAL_DIR = "/home/dhcn/thermal_project/rgb"    # Thư mục chứa ảnh thermal
# ===========================================================

# Thiết bị
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("[INFO] Sử dụng GPU")
else:
    device = torch.device('cpu')
    print("[INFO] Sử dụng CPU")

curr_gps = {"lat": TX_LAT, "lon": TX_LON}

# Load Calibration
try:
    H_final = np.load('calibration/H_matrix_1080_to_480.npy')
    data_t = np.load('calibration/mtx_thermal.npz')
    mtx_t, dist_t = data_t['mtx'], data_t['dist']
    print("[Hệ thống] Đã nạp xong các file Calibration.")
except Exception as e:
    print(f"[Lỗi] Không tìm thấy file calibration: {e}")
    exit(1)

# --- YOLOv8 TRT ---
class YOLOv8SegTRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.cfx = cuda.Device(0).make_context()
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = []
        self.host_inputs, self.cuda_inputs = [], []
        self.host_outputs, self.cuda_outputs = [], []
        for i in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(i):
                self.host_inputs.append(host_mem); self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem); self.cuda_outputs.append(cuda_mem)
        self.cfx.pop()

    def infer(self, frame_640):
        self.cfx.push()
        img = np.ascontiguousarray(cv2.cvtColor(frame_640, cv2.COLOR_BGR2RGB).transpose(2,0,1)/255.0, dtype=np.float32)
        np.copyto(self.host_inputs[0], img.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for i in range(len(self.host_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.cuda_outputs[i], self.stream)
        self.stream.synchronize()
        self.cfx.pop()
        return self.host_outputs[0].copy(), self.host_outputs[1].copy()

    def get_mask(self, out0, out1):
        try:
            if out0.size == 614400:
                protos, preds = out0.reshape(32, 120, 160), out1.reshape(37, 6300)
            else:
                preds, protos = out0.reshape(37, 6300), out1.reshape(32, 120, 160)
            conf = preds[4, :]
            idx = np.where(conf > 0.05)[0]
            if len(idx) == 0: return None
            mask_weights = preds[5:37, idx].T
            res = np.matmul(mask_weights, protos.reshape(32, -1))
            res = 1 / (1 + np.exp(-res))
            mask = np.max(res.reshape(-1, 120, 160), axis=0)
            return cv2.resize(mask, (640, 480)) > 0.5
        except:
            return None

# --- KHỞI TẠO AI ---
print("[AI] Đang nạp Model lên GPU...")
yolo = YOLOv8SegTRT("best_aerial_tree_segmentation.engine")
sgnet = SGNet(num_feats=8, kernel_size=3, scale=4).to(device).half().eval()
sgnet.load_state_dict(torch.load("weights/best_model_rmse_0.1524_epoch90.pth", map_location=device), strict=False)
print("[AI] Sẵn sàng quét rừng.")

# --- LẤY DANH SÁCH FILE VÀ GHÉP CẶP ---
def extract_number(filename):
    basename = os.path.basename(filename)
    # Tìm tất cả các cụm số, lấy cụm đầu tiên (hoặc có thể lấy cụm cuối)
    numbers = re.findall(r'\d+', basename)
    if numbers:
        return numbers[0]  # lấy số đầu tiên
    return None

# Liệt kê file
rgb_files = glob.glob(os.path.join(RGB_DIR, "*.jpg")) + glob.glob(os.path.join(RGB_DIR, "*.png")) + glob.glob(os.path.join(RGB_DIR, "*.jpeg"))
thermal_files = glob.glob(os.path.join(THERMAL_DIR, "*.tiff")) + glob.glob(os.path.join(THERMAL_DIR, "*.tif")) + glob.glob(os.path.join(THERMAL_DIR, "*.png")) + glob.glob(os.path.join(THERMAL_DIR, "*.jpg"))

print(f"Tìm thấy {len(rgb_files)} file RGB")
print(f"Tìm thấy {len(thermal_files)} file Thermal")

# Debug: in 5 file đầu
print("RGB mẫu:", rgb_files[:5])
print("Thermal mẫu:", thermal_files[:5])

# Tạo dict
rgb_dict = {}
for f in rgb_files:
    num = extract_number(f)
    if num:
        rgb_dict[num] = f
    else:
        print(f"Bỏ qua file RGB không có số: {f}")

thermal_dict = {}
for f in thermal_files:
    num = extract_number(f)
    if num:
        thermal_dict[num] = f
    else:
        print(f"Bỏ qua file Thermal không có số: {f}")

print(f"Số RGB keys: {len(rgb_dict)}")
print(f"Số Thermal keys: {len(thermal_dict)}")
print("RGB keys mẫu:", list(rgb_dict.keys())[:10])
print("Thermal keys mẫu:", list(thermal_dict.keys())[:10])

common_nums = set(rgb_dict.keys()) & set(thermal_dict.keys())
print(f"Số cặp chung: {len(common_nums)}")
if not common_nums:
    print("Không tìm thấy cặp nào! Kiểm tra lại tên file hoặc regex extract_number.")
    exit(1)

# --- VÒNG LẶP XỬ LÝ TỪNG ẢNH ---
def process_images():
    print("Hệ thống Heatmap bắt đầu xử lý ảnh từ thư mục...")
    next_event_time = time.time() + random.uniform(3, 10)
    nums = sorted(list(common_nums))
    idx = 0
    while True:
        num = nums[idx % len(nums)]
        idx += 1

        rgb_path = rgb_dict[num]
        thermal_path = thermal_dict[num]

        # Đọc ảnh RGB
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"Lỗi đọc RGB: {rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Đọc ảnh thermal
        thermal = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
        if thermal is None:
            print(f"Lỗi đọc Thermal: {thermal_path}")
            continue

        # Warp RGB
        rgb_resized = cv2.warpPerspective(rgb, H_final, (640, 480))

        # Xử lý thermal
        if len(thermal.shape) == 3:
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
        thermal_resized = cv2.resize(thermal, (160, 120))
        if thermal_resized.dtype != np.uint16:
            if thermal_resized.dtype == np.uint8:
                thermal_resized = (thermal_resized.astype(np.uint16) * 64) & 0x3FFF
            else:
                thermal_resized = thermal_resized.astype(np.uint16)

        # --- AI ---
        out0, out1 = yolo.infer(rgb_resized)
        mask = yolo.get_mask(out0, out1)

        r_in = torch.from_numpy(rgb_resized).permute(2,0,1).unsqueeze(0).to(device).half()/255.0
        t_in = torch.from_numpy(thermal_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device).half()/16383.0
        with torch.no_grad():
            sr_out = sgnet((r_in, t_in))
            if isinstance(sr_out, tuple):
                sr_out = sr_out[0]
            sr_raw = sr_out.squeeze().cpu().float().numpy() * 16383.0

        # Phát hiện hotspot
        if mask is not None:
            tree_pixels = sr_raw[mask]
            if tree_pixels.size > 50:
                mu = np.mean(tree_pixels)
                sigma = np.std(tree_pixels)
                threshold_raw = mu + (2 * sigma)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8
                )
                for i in range(1, num_labels):
                    x, y, w, h, area = stats[i]
                    if area < 50:
                        continue
                    region_mask = (labels == i)
                    region_raw_vals = sr_raw[region_mask]
                    max_raw_reg = np.max(region_raw_vals)
                    if max_raw_reg > threshold_raw:
                        temp_c = (max_raw_reg / 1000.0) + 18.8
                        now = time.time()
                        if now >= next_event_time:
                            if SIMULATE_GPS:
                                curr_gps["lat"] += random.uniform(-0.0002, 0.0002)
                                curr_gps["lon"] += random.uniform(-0.0002, 0.0002)
                            viz = rgb_resized.copy()
                            viz[mask] = [0, 0, 255]
                            _, img_enc = cv2.imencode('.jpg', viz, [cv2.IMWRITE_JPEG_QUALITY, 50])
                            payload = {'json': json.dumps({'lat': curr_gps["lat"], 'lon': curr_gps["lon"], 'temp': float(temp_c)})}
                            try:
                                requests.post(REPORT_URL, data=payload, files={'image': img_enc.tobytes()}, timeout=1)
                                print(f"[REPORT] Hotspot tại ({curr_gps['lat']:.5f}, {curr_gps['lon']:.5f}) {temp_c:.1f}°C")
                            except Exception as e:
                                print(f"Lỗi gửi báo cáo: {e}")
                            next_event_time = now + random.uniform(3, 10)

        # Giải phóng bộ nhớ
        del r_in, t_in, sr_out, sr_raw
        torch.cuda.empty_cache()
        gc.collect()

        time.sleep(0.5)

if __name__ == '__main__':
    process_images()