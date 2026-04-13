import torch
from torch2trt import torch2trt
from models.SGNet import SGNet
import torch.nn as nn

# 1. Định nghĩa lớp bọc để fix lỗi tham số
class SGNetTRTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, depth):
        # Đóng gói image và depth vào 1 tuple (x) như SGNet mong đợi
        # Chỉ lấy đầu ra đầu tiên (ảnh SR), bỏ qua đầu ra gradient cho nhẹ
        out, _ = self.model((image, depth))
        return out

# 2. Khởi tạo model gốc
device = torch.device('cuda')
model_raw = SGNet(num_feats=24, kernel_size=3, scale=4).to(device).half().eval()
model_raw.load_state_dict(torch.load("weights/best_model_rmse_0.1457_epoch70.pth"), strict=False)

# 3. Bọc model lại
model_to_convert = SGNetTRTWrapper(model_raw)

# 4. Tạo dữ liệu mẫu
x1 = torch.ones((1, 3, 480, 640)).cuda().half()
x2 = torch.ones((1, 1, 120, 160)).cuda().half()

print("Bắt đầu biên dịch sang TensorRT... (Quá trình này cực nặng)...")

# 5. Thực hiện convert (Dùng wrapper thay vì model gốc)
# max_workspace_size: Giảm xuống 1/4 RAM (1GB) để Nano không bị treo
model_trt = torch2trt(
    model_to_convert, 
    [x1, x2], 
    fp16_mode=True, 
    max_workspace_size=1 << 25
)

# 6. Lưu Engine
torch.save(model_trt.state_dict(), "weights/sgnet_trt.engine")
print("Chúc mừng! Đã tạo thành công file weights/sgnet_trt.engine")
