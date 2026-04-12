import torch.nn as nn
import torch
from torchvision import transforms

from models.modules import InvertibleConv1x1
import torch.nn.init as init
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Sequential(
        # Depthwise: Lọc từng kênh một (groups=in_channels)
        nn.Conv2d(in_channels, in_channels, kernel_size, 
                  padding=(kernel_size//2), groups=in_channels, bias=bias),
        # Pointwise: Trộn các kênh lại bằng 1x1 Conv
        nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    )


def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2),
        16: (20, 16, 2)
    }[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, bias=bias),
            act,
            conv(n_feats, n_feats, kernel_size, bias=bias)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x).mul(self.res_scale) + x


## Residual Channel Attention Block (RCAB) đã tối ưu
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias),
            CALayer(n_feat, reduction)
        )
    def forward(self, x): return self.body(x) + x

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        shared_act = nn.LeakyReLU(0.2, inplace=True)
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, act=shared_act) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x): return self.body(x) + x
 

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # SỬA TẠI ĐÂY: Dùng hàm max(1, ...) để không bao giờ ra số 0
        mid_channels = max(1, channel // reduction)
        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, mid_channels, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        self.up = up
        if up:
            self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.AdaptiveAvgPool2d(None) # Sẽ chỉnh size trong forward
            self.stride = scale
        
        self.conv = nn.Conv2d(in_channels, nr, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.up:
            x = self.upsample(x)
        else:
            # Nếu là downsample
            h, w = x.shape[2] // self.stride, x.shape[3] // self.stride
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        return self.lrelu(self.conv(x))


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3

class DownBlock(nn.Module):
    def __init__(self, scale, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        down_m = []
        for _ in range(scale):
            down_m.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.PReLU()
                )
            )
        self.downModule = nn.Sequential(*down_m)

    def forward(self, x):
        x = self.downModule(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, scale, in_channels=None, out_channels=None):
        super(UpBlock, self).__init__()
        up_m = []
        for _ in range(scale):
            up_m.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True),
                    nn.PReLU()
                )
            )
        self.downModule = nn.Sequential(*up_m)

    def forward(self, x):
        x = self.downModule(x)
        return x

class FreDiff(nn.Module):
    def __init__(self, channels,rgb_channels):
        super(FreDiff, self).__init__()

        self.fuse_c = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.fuse_sub = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(2*channels,channels,1,1,0)
        self.pre_rgb = nn.Conv2d(rgb_channels,channels,1,1,0)
        self.pre_dep = nn.Conv2d(channels,channels,1,1,0)

        self.sig = nn.Sigmoid()

    def forward(self, dp, rgb):

        dp1 = self.pre_dep(dp)
        rgb1 = self.pre_rgb(rgb)

        fuse_c = self.fuse_c(dp1)

        fuse_sub = self.fuse_sub(torch.abs(rgb1 - dp1))
        cat_fuse = torch.cat([fuse_c,fuse_sub],1)

        return self.post(cat_fuse)

class SDB(nn.Module):
    def __init__(self, channels, rgb_channels):
        super(SDB, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1)
        self.pre2 = nn.Conv2d(rgb_channels, rgb_channels, 1)
        # Sử dụng fusion đơn giản thay vì FreDiff phức tạp để cứu GFLOPS
        self.fuse = nn.Conv2d(channels + rgb_channels, channels, 1)

    def forward(self, dp, rgb):
        # Chỉ chạy FFT nếu thực sự cần, ở bản siêu nhẹ ta dùng Fusion không gian
        return self.fuse(torch.cat([dp, rgb], 1))

class get_Fre(nn.Module):
    def __init__(self):
        super(get_Fre, self).__init__()

    def forward(self, dp):

        dp = torch.fft.rfft2(dp, norm='backward')
        dp_amp = torch.abs(dp)
        dp_pha = torch.angle(dp)

        return dp_amp, dp_pha

# --- TRONG models/common.py ---
class SDM(nn.Module):
    def __init__(self, channels, rgb_channels, scale):
        super(SDM, self).__init__()
        self.rgbprocess = nn.Conv2d(rgb_channels, rgb_channels, 3, 1, 1)
        self.rgbpre = nn.Conv2d(rgb_channels, rgb_channels, 1, 1, 0)
        self.spa_process = nn.Sequential(
            InvBlock(DenseBlock, channels + rgb_channels, channels),
            nn.Conv2d(channels + rgb_channels, channels, 1, 1, 0)
        )
        self.fre_process = SDB(channels, rgb_channels)
        self.cha_att = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.fuse_process = nn.Sequential(
            InvBlock(DenseBlock, 2*channels, channels),
            nn.Conv2d(2*channels, channels, 1, 1, 0)
        )

    def forward(self, dp, rgb):
        # CHỈNH SỬA: Không gọi upBlock ở đây nữa để giữ nguyên size 120x160
        rgbpre = self.rgbprocess(rgb)
        rgb_f = self.rgbpre(rgbpre)
        
        spafuse = self.spa_process(torch.cat([dp, rgb_f], 1))
        frefuse = self.fre_process(dp, rgb_f)

        cat_f = self.fuse_process(torch.cat([spafuse, frefuse], 1))
        cha_res = self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f
        
        out = cha_res + dp
        # CHỈNH SỬA: Không gọi downBlock ở đây
        return out, rgbpre
    
class Get_gradient_nopadding_rgb(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32).view(1,1,3,3)
        kernel_h = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('weight_h', kernel_h)
        self.register_buffer('weight_v', kernel_v)

    def forward(self, x):
        def calc(c):
            v = F.conv2d(c, self.weight_v.type_as(x), padding=1)
            h = F.conv2d(c, self.weight_h.type_as(x), padding=1)
            return torch.sqrt(v*v + h*h + 1e-6)
        return torch.cat([calc(x[:, i:i+1]) for i in range(3)], dim=1)
    

class Get_gradient_nopadding_d(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding_d, self).__init__()
        kernel_v = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_h = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('weight_h', kernel_h)
        self.register_buffer('weight_v', kernel_v)

    def forward(self, x):
        # x thường là ảnh nhiệt 1 kênh
        v = F.conv2d(x, self.weight_v.type_as(x), padding=1)
        h = F.conv2d(x, self.weight_h.type_as(x), padding=1)
        return torch.sqrt(torch.pow(v, 2) + torch.pow(h, 2) + 1e-6)

class GCM(nn.Module):
    def __init__(self,n_feats,scale):
        super(GCM, self).__init__()
        self.grad_rgb = Get_gradient_nopadding_rgb()
        self.grad_d = Get_gradient_nopadding_d()
        self.upBlock = DenseProjection(1, 1, scale, up=True, bottleneck=False)
        self.downBlock = DenseProjection(n_feats, n_feats, scale, up=False, bottleneck=False)
        self.c_rgb = default_conv(3,n_feats,3)
        self.c_d = default_conv(1,n_feats,3)
        self.c_fuse = default_conv(n_feats,n_feats,3)

        self.rg_d = ResidualGroup(default_conv, n_feats, 3, reduction=16, n_resblocks=4)
        self.rb_rgbd = ResBlock(default_conv, n_feats, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.fuse_process = nn.Sequential(nn.Conv2d(2*n_feats, n_feats, 1, 1, 0),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4))
        self.re_g = default_conv(n_feats,1,3)
        self.re_d = default_conv(n_feats,1,3)
        self.c_sab = default_conv(1,n_feats,3)
        self.sig = nn.Sigmoid()
        self.d1 = nn.Sequential(default_conv(1,n_feats,3),
                                ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=8))

        self.CA = CALayer(n_feats,reduction=4)

        grad_conv = [
            default_conv(1, n_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(n_feats, n_feats, kernel_size=3, bias=True),
        ]
        self.grad_conv = nn.Sequential(*grad_conv)
        self.grad_rg = nn.Sequential(ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4),
        ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4))

    def forward(self, depth,rgb):

        depth = self.upBlock(depth)

        grad_rgb = self.grad_rgb(rgb)
        grad_d = self.grad_d(depth)

        rgb1 = self.c_rgb(grad_rgb)
        d1 = self.c_d(grad_d)

        rgb2 = self.rb_rgbd(rgb1)
        d2 = self.rg_d(d1)

        cat1 = torch.cat([rgb2,d2],dim=1)

        inn1 = self.fuse_process(cat1)

        d3 = d1 + self.CA(inn1)

        grad_d2 = self.c_fuse(d3)

        out_re = self.re_g(grad_d2)

        d4 = self.d1(depth)

        grad_d3 = self.grad_conv(out_re) + d4

        grad_d4 = self.grad_rg(grad_d3)

        return out_re,self.downBlock(grad_d4)
