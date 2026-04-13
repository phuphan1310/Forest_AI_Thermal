import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import base64
import json
import time
import torch
import torch.nn as nn
import gc
import threading
from flask import Flask, jsonify
from models.SGNet import SGNet

app = Flask(__name__)
device = torch.device('cuda')

# --- BIẾN TOÀN CỤC ---
latest_f_r = None          # ảnh RGB đã warp (640x480, dùng cho AI)
latest_f_r_raw = None      # ảnh RGB gốc (1920x1080, dùng để hiển thị)
latest_f_t = None
ai_result = {"status": "Đang khởi động..."}
frame_lock = threading.Lock()
result_lock = threading.Lock()

# Load Calibration cho độ phân giải 1920x1080
data_r = np.load('calibration/mtx_rgb_1080.npz')
mtx_r = data_r['mtx']
dist_r = data_r['dist']
data_t = np.load('calibration/mtx_thermal.npz')
mtx_t, dist_t = data_t['mtx'], data_t['dist']
H_final = np.load('calibration/H_matrix_1080_to_480.npy')   # Ma trận homography 1080p -> 640x480

# --- YOLO TRT ---
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
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
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
        if out0.size == 614400:
            protos, preds = out0.reshape(32, 120, 160), out1.reshape(37, 6300)
        else:
            preds, protos = out0.reshape(37, 6300), out1.reshape(32, 120, 160)
        conf = preds[4, :]
        idx = np.where(conf > 0.05)[0]
        if len(idx) == 0:
            return None
        mask_weights = preds[5:, idx].T
        res = np.matmul(mask_weights, protos.reshape(32, -1))
        res = 1 / (1 + np.exp(-res))
        return cv2.resize(np.max(res.reshape(-1, 120, 160), axis=0), (640, 480)) > 0.5

# --- KHỞI TẠO AI ---
print("Đang nạp AI...")
try:
    yolo = YOLOv8SegTRT("best_aerial_tree_segmentation.engine")
    print("YOLO TensorRT loaded successfully")
except Exception as e:
    print(f"Lỗi load YOLO: {e}")
    yolo = None

try:
    sgnet = SGNet(num_feats=8, kernel_size=3, scale=4).to(device).half().eval()
    sgnet.load_state_dict(torch.load("weights/best_model_rmse_0.1524_epoch90.pth", map_location=device), strict=False)
    print("SGNet loaded successfully")
except Exception as e:
    print(f"Lỗi load SGNet: {e}")
    sgnet = None

def encode(img, q=40):
    if img is None or img.size == 0:
        return ""
    _, b = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return base64.b64encode(b).decode('utf-8')

# --- LUỒNG CAMERA (RGB + Thermal 14-bit) ---
def camera_loop():
    global latest_f_r, latest_f_r_raw, latest_f_t

    # RGB: pipeline GStreamer (1920x1080)
    GST_RGB = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    cap_rgb = cv2.VideoCapture(GST_RGB, cv2.CAP_GSTREAMER)
    if not cap_rgb.isOpened():
        print("[Camera] Không thể mở camera RGB. Dùng ảnh giả.")
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_thermal = np.zeros((120, 160), dtype=np.uint16)
        while True:
            with frame_lock:
                latest_f_r = dummy_rgb
                latest_f_r_raw = dummy_rgb
                latest_f_t = dummy_thermal
            time.sleep(0.05)
        return

    print("[Camera] Camera RGB đã mở (GStreamer).")

    # Thermal: PureThermal (FLIR Lepton 3.5) qua UVC
    thermal_cap = None
    try:
        thermal_cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        if not thermal_cap.isOpened():
            raise Exception("Không mở được thermal device")
        thermal_cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        thermal_cap.set(cv2.CAP_PROP_MONOCHROME, 1)
        ret, test = thermal_cap.read()
        if not ret:
            raise Exception("Đọc thử thermal thất bại")
        print(f"[Camera] Thermal đã mở, dtype mẫu: {test.dtype}, shape: {test.shape}")
        thermal_cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    except Exception as e:
        print(f"[Camera] Lỗi camera nhiệt: {e}. Dùng ảnh giả.")
        thermal_cap = None

    frame_count = 0
    while True:
        ret_r, frame_rgb = cap_rgb.read()
        if not ret_r:
            print("[Camera] Lỗi đọc RGB, giữ nguyên frame cũ.")
            time.sleep(0.01)
            continue

        # Lưu ảnh RGB gốc (1920x1080) để hiển thị
        with frame_lock:
            latest_f_r_raw = frame_rgb.copy()

        # Áp dụng homography để chuyển 1920x1080 -> 640x480 (căn chỉnh)
        rgb_aligned = cv2.warpPerspective(frame_rgb, H_final, (640, 480))

        # Đọc thermal
        if thermal_cap is not None:
            ret_t, frame_thermal = thermal_cap.read()
            if not ret_t:
                frame_thermal = np.zeros((120, 160), dtype=np.uint16)
        else:
            frame_thermal = np.zeros((120, 160), dtype=np.uint16)

        # Resize thermal về 160x120 nếu cần
        if frame_thermal.shape != (120, 160):
            frame_thermal = cv2.resize(frame_thermal, (160, 120), interpolation=cv2.INTER_NEAREST)

        # Đảm bảo frame_thermal là uint16 (14-bit, max 16383)
        if frame_thermal.dtype != np.uint16:
            if frame_thermal.dtype == np.uint8:
                frame_thermal = (frame_thermal.astype(np.uint16) * 64) & 0x3FFF
            else:
                frame_thermal = frame_thermal.astype(np.uint16)

        with frame_lock:
            latest_f_r = rgb_aligned          # Đã warp về 640x480 (dùng cho AI)
            latest_f_t = frame_thermal

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[Camera] Đã xử lý {frame_count} frame (RGB + Thermal 14-bit).")

        time.sleep(0.01)

threading.Thread(target=camera_loop, daemon=True).start()

# --- LUỒNG AI NGẦM ---
# --- LUỒNG AI NGẦM ---
def ai_loop():
    global ai_result
    print("Luồng AI bắt đầu chạy ngầm...")
    while True:
        start_t = time.time()
        if latest_f_r is not None and yolo is not None and sgnet is not None:
            with frame_lock:
                f_r = latest_f_r.copy()           
                f_t = latest_f_t.copy()
                f_r_raw = latest_f_r_raw.copy() if latest_f_r_raw is not None else None
            try:
                # Tiền xử lý
                if len(f_t.shape) == 3:
                    f_t = cv2.cvtColor(f_t, cv2.COLOR_BGR2GRAY)
                t_f = cv2.undistort(f_t, mtx_t, dist_t)
                rgb_aligned = f_r                  
                t_120 = cv2.resize(t_f, (160, 120))

                # YOLO
                o0, o1 = yolo.infer(rgb_aligned)
                mask = yolo.get_mask(o0, o1)

                # SGNet
                r_in = torch.from_numpy(rgb_aligned).permute(2,0,1).unsqueeze(0).to(device).half()/255.0
                t_in = torch.from_numpy(t_120.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device).half()/16383.0
                with torch.no_grad():
                    sr_out = sgnet((r_in, t_in))
                    if isinstance(sr_out, tuple):
                        sr_out = sr_out[0]
                    sr_raw = sr_out.squeeze().cpu().float().numpy() * 16383.0

                # Tạo ảnh hiển thị
                sr_8bit = cv2.normalize(sr_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                t8_sr = cv2.applyColorMap(sr_8bit, cv2.COLORMAP_JET)

                yolo_mask_viz = rgb_aligned.copy()
                if mask is not None:
                    yolo_mask_viz[mask] = [0, 0, 255]

                # --- PHẦN TÍNH TOÁN THỐNG KÊ VÀ HIỆN TEXT THÔNG MINH ---
                if mask is not None:
                    # Lấy tất cả giá trị raw của vùng CÂY để tính toán ngưỡng động
                    tree_pixels = sr_raw[mask]
                    if tree_pixels.size > 100: # Đảm bảo đủ mẫu để tính toán
                        mu = np.mean(tree_pixels)
                        sigma = np.std(tree_pixels)
                        # Ngưỡng: Chỉ những pixel nóng hơn mức trung bình 2 lần độ lệch chuẩn
                        threshold_raw = mu + (2 * sigma)

                        # Tìm các vùng cây riêng biệt
                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                            mask.astype(np.uint8), connectivity=8
                        )

                        for i in range(1, num_labels):
                            x, y, w, h, area = stats[i]
                            if area < 50: continue
                            
                            region_mask = (labels == i)
                            region_raw_vals = sr_raw[region_mask]
                            max_raw_reg = np.max(region_raw_vals)

                            # ĐIỀU KIỆN: Chỉ hiện text nếu vùng này nóng bất thường so với tổng thể rừng
                            if max_raw_reg > threshold_raw:
                                # Tính Celsius (dùng công thức 1000 như bạn đã chọn)
                                temp_c = (max_raw_reg / 1000.0) + 18.8
                                
                                # Vẽ Box và Text
                                cv2.rectangle(t8_sr, (x, y), (x+w, y+h), (0, 255, 255), 2)
                                text = f"{temp_c:.1f}C"
                                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(t8_sr, (x, y-th-5), (x+tw, y), (0, 0, 0), -1)
                                cv2.putText(t8_sr, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # --- KẾT THÚC PHẦN THÊM ---

                with result_lock:
                    ai_result = {
                        "rgb_raw": encode(f_r_raw, 30) if f_r_raw is not None else "",
                        "t8_raw": encode(cv2.applyColorMap(cv2.normalize(t_120, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET), 30),
                        "rgb_aligned": encode(rgb_aligned, 40),
                        "yolo_mask": encode(yolo_mask_viz, 50),
                        "sgnet_hr": encode(t8_sr, 50),
                        "max_temp": round(float((np.max(sr_raw)/1000.0)+18.8), 1)
                    }
                print("[AI] Đã cập nhật kết quả")
                del r_in, t_in, sr_out, sr_raw, rgb_aligned
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"[AI] Lỗi: {e}")
                import traceback
                traceback.print_exc()
        else:
            time.sleep(0.1)
        time.sleep(max(0.1, 1.0 - (time.time() - start_t)))
        
threading.Thread(target=ai_loop, daemon=True).start()

@app.route('/process_edge', methods=['GET'])
def process_edge():
    with result_lock:
        return jsonify(ai_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=False)