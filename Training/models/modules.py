import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
import models.thops as thops

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            # THAY ĐỔI 1: Chỉ tính logdet khi đang ở chế độ Train
            dlogdet = None
            if self.training:
                pixels = thops.pixels(input)
                # Tính toán trên float32 để tránh lỗi với FP16
                dlogdet = torch.slogdet(self.weight.float())[1] * pixels
            
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                # Ép sang double để nghịch đảo chính xác, sau đó ép về kiểu của input (half/float)
                weight = torch.inverse(self.weight.double()).type_as(self.weight)\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            # THAY ĐỔI 2: Đảm bảo các buffer luôn cùng device và kiểu dữ liệu với input
            self.p = self.p.to(input.device).type_as(input)
            self.sign_s = self.sign_s.to(input.device).type_as(input)
            self.l_mask = self.l_mask.to(input.device).type_as(input)
            self.eye = self.eye.to(input.device).type_as(input)
            
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            
            dlogdet = None
            if self.training:
                dlogdet = thops.sum(self.log_s) * thops.pixels(input)
                
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l_inv = torch.inverse(l.double()).type_as(l)
                u_inv = torch.inverse(u.double()).type_as(u)
                w = torch.matmul(u_inv, torch.matmul(l_inv, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            # THAY ĐỔI 3: Chỉ cộng logdet nếu nó tồn tại (giúp tiết kiệm RAM khi Infer)
            if logdet is not None and dlogdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None and dlogdet is not None:
                logdet = logdet - dlogdet
            return z, logdet