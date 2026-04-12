import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import * # Đảm bảo file common.py có các class ResBlock, ResidualGroup, SDM, GCM

class SGNet(nn.Module):
    def __init__(self, num_feats=8, kernel_size=3, scale=4):
        super(SGNet, self).__init__()
        
        # 1. KHAI BÁO THIẾU: Hàm kích hoạt dùng chung
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 2. Nhánh xử lý ảnh RGB (Guidance)
        self.conv_rgb1 = nn.Conv2d(3, num_feats, kernel_size, padding=1)
        self.rgb_rb2 = ResBlock(default_conv, num_feats, kernel_size)

        # 3. Nhánh xử lý ảnh Nhiệt (LR)
        self.conv_dp1 = nn.Conv2d(1, num_feats, kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, 16, n_resblocks=1)

        # 4. Nhánh Gradient (Trích xuất cạnh)
        self.gradNet = GCM(n_feats=num_feats, scale=scale)

        # 5. Lớp gộp (Fusion) dữ liệu Cạnh và dữ liệu Nhiệt
        # Khai báo nn.Conv2d ở đây để đảm bảo tính toán nhanh
        self.c_grad = nn.Conv2d(2*num_feats, num_feats, 1)

        # 6. Cây cầu hòa trộn thông tin màu và nhiệt
        self.bridge1 = SDM(channels=num_feats, rgb_channels=num_feats, scale=scale)

        # 7. Khối Tail (Xử lý sâu ở vùng ảnh thấp 160x120)
        self.tail = ResidualGroup(default_conv, num_feats, kernel_size, 16, n_resblocks=1)

        # 8. Bộ phóng đại (Upsampler) - Bilinear để tiết kiệm GFLOPS
        self.upsampler = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        
        # 9. Lớp tích chập cuối cùng để trả về ảnh 1 kênh (Nhiệt HR)
        self.last_conv = nn.Conv2d(num_feats, 1, kernel_size=3, padding=1)
        
        # 10. Shortcut Bicubic (Phần cộng bù sai số)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

    def forward(self, x):
        # x là tuple gồm (ảnh_màu, ảnh_nhiệt_lr)
        image, depth = x

        # BƯỚC 1: Lấy thông tin cạnh từ GCM
        out_re, grad_d = self.gradNet(depth, image) # Kết quả ở vùng 160x120

        # BƯỚC 2: Xử lý nhánh nhiệt
        dp = self.act(self.conv_dp1(depth))
        dp = self.dp_rg1(dp)

        # BƯỚC 3: Xử lý nhánh màu và hạ độ phân giải để đồng bộ không gian
        rgb = self.act(self.conv_rgb1(image))
        rgb = self.rgb_rb2(rgb)
        # Ép RGB về 160x120 để thực hiện Fusion nhẹ
        rgb_s = F.interpolate(rgb, scale_factor=1/4, mode='bilinear', align_corners=False)

        # BƯỚC 4: Fusion Cạnh + Nhiệt + Màu
        dp_combined = self.c_grad(torch.cat([dp, grad_d], 1))
        ca_in, _ = self.bridge1(dp_combined, rgb_s)
        
        # BƯỚC 5: Xử lý Tail-end
        out_low = self.tail(ca_in)
        
        # BƯỚC 6: Phóng đại lên 640x480
        out_high = self.upsampler(out_low)
        out = self.last_conv(out_high)
        
        # BƯỚC 7: Cộng Residual (Bicubic)
        out = out + self.bicubic(depth)

        return out, out_re