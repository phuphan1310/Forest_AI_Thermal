from flask import Flask, render_template, request, jsonify
import os, json, time
from waitress import serve

app = Flask(__name__)

# Cấu hình lưu trữ ảnh báo cáo
UPLOAD_FOLDER = 'static/detections'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cơ sở dữ liệu tạm thời lưu danh sách cây khô phát hiện được
tree_database = [] 

# --- 1. NHẬN BÁO CÁO AI TỪ JETSON ---
@app.route('/report', methods=['POST'])
def report():
    try:
        # Nhận ảnh và dữ liệu JSON từ Jetson
        img_file = request.files['image']
        data = json.loads(request.form['json'])
        
        # Lưu ảnh vào thư mục static
        filename = f"tree_{int(time.time())}_{data.get('lat', 0)}.jpg"
        img_file.save(os.path.join(UPLOAD_FOLDER, filename))
        
        # Thêm dữ liệu vào database để vẽ heatmap
        tree_database.append({
            "lat": data['lat'], 
            "lon": data['lon'],
            "temp": round(data['temp'], 1), 
            "img": filename,
            "time": time.strftime('%H:%M:%S')
        })
        
        print(f"[AI REPORT] Phát hiện cây nóng: {data['temp']}°C tại {data['lat']}, {data['lon']}")
        return "OK", 200
    except Exception as e:
        print(f"[LỖI REPORT] {e}")
        return "Error", 500

# --- 2. CUNG CẤP DỮ LIỆU CHO BẢN ĐỒ LEAFLET ---
@app.route('/get_data')
def get_data():
    # Trình duyệt sẽ gọi endpoint này mỗi 3 giây để vẽ lại Heatmap
    return jsonify(tree_database)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("--- SERVER HEATMAP ĐANG CHẠY TẠI PORT 5001 ---")
    print("--- CHẾ ĐỘ: CHỈ NHẬN DỮ LIỆU TỌA ĐỘ VÀ ẢNH BÁO CÁO ---")
    # Chạy server chuyên nghiệp với Waitress
    serve(app, host='0.0.0.0', port=5001, threads=10)