import torch
from thop import profile
from models.SGNet import SGNet # Import model của bạn

def measure():
    # 1. Khởi tạo model
    num_feats = 8 # Hoặc 24, 32 tùy bản bạn đang dùng
    model = SGNet(num_feats=num_feats, kernel_size=3, scale=4)
    
    # 2. Tạo dữ liệu giả (Dummy Input) đúng kích thước thực tế
    # Batch size = 1
    input_rgb = torch.randn(1, 3, 480, 640)
    input_thermal = torch.randn(1, 1, 120, 160)
    inputs = (input_rgb, input_thermal)

    # 3. Sử dụng thop để đo
    # macs: Multiply-Accumulate Operations (thường 1 MAC ~ 2 FLOPs)
    # params: Tổng số tham số
    macs, params = profile(model, inputs=(inputs, ), verbose=False)

    # 4. Tính toán con số
    flops = macs * 2 # Ước tính FLOPs
    
    print("-" * 30)
    print(f"Cấu hình Model: num_feats={num_feats}")
    print(f"Số lượng Parameters: {params / 1e6:.2f} Million")
    print(f"Số lượng MACs: {macs / 1e9:.2f} G")
    print(f"Số lượng FLOPs: {flops / 1e9:.2f} GFLOPS")
    print("-" * 30)

if __name__ == "__main__":
    measure()