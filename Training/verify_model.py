import torch
from models.SGNet import SGNet
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_model_structure():
    # 1. Khởi tạo model với thông số bạn định dùng
    num_feats = 16
    scale = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"--- Đang kiểm tra cấu hình: Feats={num_feats}, Scale={scale} ---")
    
    try:
        model = SGNet(num_feats=num_feats, kernel_size=3, scale=scale).to(device)
    except Exception as e:
        print(f"Lỗi khởi tạo model: {e}")
        return

    # 2. KIỂM TRA SỐ LƯỢNG RESBLOCKS
    print("\n[1] KIỂM TRA SỐ LƯỢNG RESBLOCKS (Mục tiêu: mỗi nhóm chỉ có 2)")
    
    # Kiểm tra nhóm Low-Res (dp_rg1)
    lr_blocks = len([m for m in model.dp_rg1.body if "RCAB" in str(type(m))])
    print(f"  + Số blocks trong dp_rg1: {lr_blocks} (Mục tiêu: 2)")
    
    # Kiểm tra nhóm Tail
    tail_blocks = 0
    for group in model.tail:
        tail_blocks += len([m for m in group.body if "RCAB" in str(type(m))])
    print(f"  + Tổng số blocks trong Tail: {tail_blocks} (Mục tiêu: 6 - vì 3 nhóm x 2)")

    # 3. KIỂM TRA TỔNG THAM SỐ (Dung lượng)
    params = count_parameters(model)
    estimated_size_mb = (params * 4) / (1024 * 1024) # Giả định Float32 (4 bytes)
    
    print("\n[2] KIỂM TRA DUNG LƯỢNG")
    print(f"  + Tổng số tham số: {params:,}")
    print(f"  + Dung lượng file dự kiến (.pth): {estimated_size_mb:.2f} MB")
    
    if estimated_size_mb < 20:
        print("  => KẾT QUẢ: MÔ HÌNH ĐÃ NHẸ (ĐẠT CHUẨN JETSON NANO)")
    else:
        print("  => KẾT QUẢ: MÔ HÌNH VẪN NẶNG (CẦN KIỂM TRA LẠI N_LR VÀ N_TAIL)")

    # 4. KIỂM TRA THỨ TỰ XỬ LÝ (Forward Flow)
    print("\n[3] KIỂM TRA THỨ TỰ LỚP (Vị trí Tail và Upsampler)")
    # Lấy danh sách các module con theo thứ tự khai báo
    children = list(model.named_children())
    child_names = [c[0] for c in children]
    
    # Tìm vị trí của tail và upsampler trong danh sách khai báo
    try:
        idx_tail = child_names.index('tail')
        idx_upsampler = child_names.index('upsampler')
        
        if idx_tail < idx_upsampler:
            print("  + Thứ tự khai báo: Tail đứng TRƯỚC Upsampler (Chuẩn tối ưu RAM)")
        else:
            print("  + Thứ tự khai báo: Tail đứng SAU Upsampler (Cấu hình cũ tốn RAM)")
    except:
        print("  + Không tìm thấy tên lớp 'tail' hoặc 'upsampler'.")

    # 5. CHẠY THỬ (DUMMY INFERENCE)
    print("\n[4] CHẠY THỬ ĐỂ KIỂM TRA LỖI")
    dummy_rgb = torch.randn(1, 3, 480, 640).to(device)
    dummy_t = torch.randn(1, 1, 120, 160).to(device)
    
    try:
        with torch.no_grad():
            out, _ = model((dummy_rgb, dummy_t))
        print(f"  + Chạy thử thành công! Output shape: {out.shape}")
        if out.shape == (1, 1, 480, 640):
            print("  + Kích thước đầu ra chính xác 640x480.")
    except Exception as e:
        print(f"  + Lỗi khi chạy thử: {e}")

if __name__ == "__main__":
    check_model_structure()