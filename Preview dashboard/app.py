from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import torch
import os
import requests
from ultralytics import YOLO
from models.sgnet_arch import SGNet

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# IP Tailscale của Jetson Nano - THAY ĐỔI THEO IP THỰC TẾ CỦA BẠN
JETSON_NANO_IP = "100.72.19.8" 

# --- LOAD MODELS CHO PHẦN 1 & 2 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('models/best_aerial_tree_segmentation.pt')
sgnet_model = SGNet(num_feats=32, kernel_size=3, scale=4).to(device).eval()
sgnet_model.load_state_dict(torch.load('models/best_model_rmse_0.0073_epoch98.pth', map_location=device))

@app.route('/')
def index():
    return render_template('index.html')

# --- ROUTE CHO PHẦN 1 ---
@app.route('/yolo_infer', methods=['POST'])
def yolo_infer():
    file = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, 'yolo_in.jpg')
    file.save(path)
    results = yolo_model.predict(path, save=False)
    res_plotted = results[0].plot()
    out_path = os.path.join(UPLOAD_FOLDER, 'yolo_out.jpg')
    cv2.imwrite(out_path, res_plotted)
    return jsonify({"output_url": "/static/uploads/yolo_out.jpg"})

# --- ROUTE CHO PHẦN 2 ---
@app.route('/sgnet_infer', methods=['POST'])
def sgnet_infer():
    try:
        rgb_file = request.files['rgb']
        thermal_file = request.files['thermal']
        rgb_path = os.path.join(UPLOAD_FOLDER, 'sg_rgb.jpg')
        t_path = os.path.join(UPLOAD_FOLDER, 'sg_t_raw.tiff')
        rgb_file.save(rgb_path)
        thermal_file.save(t_path)
        
        img_rgb = cv2.imread(rgb_path)
        img_t = cv2.imread(t_path, cv2.IMREAD_UNCHANGED)
        if img_rgb is None or img_t is None: return jsonify({"error": "Lỗi đọc file"}), 400

        if len(img_t.shape) == 3: img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
        img_rgb_640 = cv2.resize(img_rgb, (640, 480))
        img_t_160 = cv2.resize(img_t, (160, 120), interpolation=cv2.INTER_AREA)
        
        # --- CHẠY SGNET ---
        rgb_ts = torch.from_numpy(img_rgb_640).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
        denom = 16383.0 if img_t.dtype == np.uint16 else 255.0
        t_ts = torch.from_numpy(img_t_160.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device) / denom
        
        with torch.no_grad():
            outputs = sgnet_model((rgb_ts, t_ts))
            sr_out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            sr_img_gray = (sr_out.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        sr_color = cv2.applyColorMap(sr_img_gray, cv2.COLORMAP_JET)

        # --- TẠO ẢNH ZOOM 800% ---
        # Lấy vùng trung tâm 40x40
        p = 20; cy, cx = 240, 320
        # Ảnh gốc trước SR (Bicubic up)
        lr_up = cv2.resize(img_t_160, (640, 480), interpolation=cv2.INTER_NEAREST)
        if lr_up.dtype == np.uint16: lr_up = (lr_up / 64).astype(np.uint8)
        lr_color = cv2.applyColorMap(lr_up, cv2.COLORMAP_JET)

        # Zoom 800%
        zoom_lr = cv2.resize(lr_color[cy-p:cy+p, cx-p:cx+p], (320, 320), interpolation=cv2.INTER_NEAREST)
        zoom_sr = cv2.resize(sr_color[cy-p:cy+p, cx-p:cx+p], (320, 320), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'sg_out.jpg'), sr_color)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'zoom_lr.jpg'), zoom_lr)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'zoom_sr.jpg'), zoom_sr)
        
        return jsonify({
            "output_url": "/static/uploads/sg_out.jpg",
            "zoom_lr_url": "/static/uploads/zoom_lr.jpg",
            "zoom_sr_url": "/static/uploads/zoom_sr.jpg"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --- ROUTE CHO PHẦN 3: KẾT NỐI JETSON NANO ---
@app.route('/trigger_jetson', methods=['POST'])
def trigger_jetson():
    try:
        # Gọi Jetson lấy dữ liệu đã xử lý sẵn trong RAM
        # Timeout 10s là cực kỳ dư dả vì Jetson trả kết quả < 0.1s
        response = requests.get(f"http://{JETSON_NANO_IP}:5001/process_edge", timeout=10)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": f"Lỗi kết nối Jetson: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)