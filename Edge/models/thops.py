import torch

def onehot(y, num_classes):
    # TỐI ƯU: Dùng .type_as(y) để tự động khớp Float32 hoặc Float16 và cùng Device
    y_onehot = torch.zeros(y.size(0), num_classes).type_as(y)
    if len(y.size()) == 1:
        y_onehot = y_onehot.scatter_(1, y.unsqueeze(-1).long(), 1)
    elif len(y.size()) == 2:
        y_onehot = y_onehot.scatter_(1, y.long(), 1)
    else:
        raise ValueError("[onehot]: y should be in shape [B], or [B, C]")
    return y_onehot


def sum(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sum(tensor)
    else:
        # TỐI ƯU: PyTorch hiện đại hỗ trợ truyền tuple vào dim, 
        # giúp tránh vòng lặp for làm chậm và tốn bộ nhớ đệm.
        if isinstance(dim, int):
            return torch.sum(tensor, dim=dim, keepdim=keepdim)
        else:
            return torch.sum(tensor, dim=tuple(dim), keepdim=keepdim)


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.mean(tensor)
    else:
        # TỐI ƯU: Tương tự hàm sum, dùng tuple cho dim
        if isinstance(dim, int):
            return torch.mean(tensor, dim=dim, keepdim=keepdim)
        else:
            return torch.mean(tensor, dim=tuple(dim), keepdim=keepdim)


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    Sử dụng slicing (:) trong PyTorch là phép toán 'view', 
    không tạo bản sao dữ liệu trong RAM -> Cực kỳ tiết kiệm cho Jetson.
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_feature(tensor_a, tensor_b):
    # Lưu ý: torch.cat sẽ tạo ra một tensor mới, chiếm thêm RAM.
    # Hãy đảm bảo xóa tensor cũ ngay sau khi cat nếu không dùng tới.
    return torch.cat((tensor_a, tensor_b), dim=1)


def pixels(tensor):
    # Trả về số lượng pixel trong 1 channel (H * W)
    return int(tensor.size(2) * tensor.size(3))