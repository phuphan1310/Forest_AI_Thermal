# data/flir_dataloader.py
from torchvision import transforms
import numpy as np
import os
import cv2
from glob import glob
from torch.utils.data import Dataset
import torch
import random

class FLIR_Dataset(Dataset):
    def __init__(self, root_dir, scale=4, transform=None, train=True, max_samples=None):
        self.transform = transform
        self.scale = scale
        self.max_samples = max_samples
        
        # Kích thước patch cố định
        self.gt_h, self.gt_w = 128, 160  # GT: 128x160
        self.lr_h, self.lr_w = self.gt_h // scale, self.gt_w // scale  # LR: 32x40
        
        # Kích thước ảnh gốc sau khi resize RGB
        self.rgb_h, self.rgb_w = 520, 640  # Resize RGB về 640x520
        
        split = 'train' if train else 'val'
        data_dir = os.path.join(root_dir, split)
        
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.gt_dir = os.path.join(data_dir, 'gt')
        self.train = train
        
        # Lấy danh sách file
        rgb_files = glob(os.path.join(self.rgb_dir, '*.jpg'))
        depth_files = glob(os.path.join(self.depth_dir, '*.jpeg'))
        gt_files = glob(os.path.join(self.gt_dir, '*.jpeg'))
        
        # Tìm tên chung
        rgb_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in rgb_files}
        depth_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in depth_files}
        gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
        
        common_names = set(rgb_dict.keys()) & set(depth_dict.keys()) & set(gt_dict.keys())
        common_names = sorted(list(common_names))
        
        # Giới hạn số lượng samples
        if max_samples is not None and len(common_names) > max_samples:
            if train:
                common_names = random.sample(common_names, max_samples)
            else:
                common_names = common_names[:max_samples]
        
        self.rgb_files = [rgb_dict[name] for name in common_names]
        self.depth_files = [depth_dict[name] for name in common_names]
        self.gt_files = [gt_dict[name] for name in common_names]
        
        # Tính các vị trí top, left hợp lệ (chia hết cho scale)
        self.valid_tops = list(range(0, self.rgb_h - self.gt_h + 1, self.scale))
        self.valid_lefts = list(range(0, self.rgb_w - self.gt_w + 1, self.scale))
        
        print(f"\n=== {split} set ===")
        print(f"Total files: {len(common_names)}")
        print(f"GT patch size: {self.gt_h}x{self.gt_w}")
        print(f"LR patch size: {self.lr_h}x{self.lr_w}")
        print(f"RGB size after resize: {self.rgb_h}x{self.rgb_w}")
        print(f"Valid top positions (step {self.scale}): {len(self.valid_tops)} positions")
        print(f"Valid left positions (step {self.scale}): {len(self.valid_lefts)} positions")
        print(f"Total possible patches per image: {len(self.valid_tops) * len(self.valid_lefts)}")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Đọc ảnh
        rgb = cv2.imread(self.rgb_files[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(self.depth_files[idx], cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(self.gt_files[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize RGB về 640x520
        rgb = cv2.resize(rgb, (self.rgb_w, self.rgb_h))
        
        # Resize depth và GT về cùng kích thước (nếu cần)
        # Giả sử depth và GT đã cùng size 640x512, cần resize GT lên 640x520?
        # Nhưng GT gốc là 640x512, cần pad thêm hoặc crop?
        
        # Cách đơn giản: Resize depth và GT về 640x520
        gt = cv2.resize(gt, (self.rgb_w, self.rgb_h))
        depth = cv2.resize(depth, (self.rgb_w, self.rgb_h))
        
        # Chọn vị trí patch ngẫu nhiên (chia hết cho scale)
        if self.train:
            top = random.choice(self.valid_tops)
            left = random.choice(self.valid_lefts)
        else:
            # Center crop cho validation
            top = self.valid_tops[len(self.valid_tops) // 2]
            left = self.valid_lefts[len(self.valid_lefts) // 2]
        
        # Cắt patch trên GT
        gt_patch = gt[top:top+self.gt_h, left:left+self.gt_w]
        
        # Cắt patch trên RGB (cùng vị trí)
        rgb_patch = rgb[top:top+self.gt_h, left:left+self.gt_w, :]
        
        # Tính vị trí tương ứng trên LR (chia tọa độ cho scale)
        lr_top = top // self.scale
        lr_left = left // self.scale
        
        # Cắt patch trên depth (ở kích thước LR)
        depth_lr = cv2.resize(depth, (self.rgb_w // self.scale, self.rgb_h // self.scale))
        lr_patch = depth_lr[lr_top:lr_top+self.lr_h, lr_left:lr_left+self.lr_w]
        
        # Normalize
        rgb_patch = rgb_patch.astype(np.float32) / 255.0
        lr_patch = lr_patch.astype(np.float32) / 255.0
        gt_patch = gt_patch.astype(np.float32) / 255.0
        
        # Transform
        if self.transform:
            rgb_patch = self.transform(rgb_patch)
            lr_patch = torch.from_numpy(lr_patch).unsqueeze(0).float()
            gt_patch = torch.from_numpy(gt_patch).unsqueeze(0).float()
        else:
            rgb_patch = torch.from_numpy(np.transpose(rgb_patch, (2, 0, 1))).float()
            lr_patch = torch.from_numpy(lr_patch).unsqueeze(0).float()
            gt_patch = torch.from_numpy(gt_patch).unsqueeze(0).float()

        # In ra để kiểm tra (chỉ in vài mẫu đầu)
        if idx < 3:
            print(f"\nSample {idx}:")
            print(f"  Top: {top}, Left: {left}")
            print(f"  LR Top: {lr_top}, LR Left: {lr_left}")
            print(f"  RGB patch: {rgb_patch.shape}")
            print(f"  LR patch: {lr_patch.shape}")
            print(f"  GT patch: {gt_patch.shape}")

        return {
            'guidance': rgb_patch,
            'lr': lr_patch,
            'gt': gt_patch
        }