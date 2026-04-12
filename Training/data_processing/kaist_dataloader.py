# data/kaist_dataloader.py
from torchvision import transforms
import numpy as np
import os
import cv2
from glob import glob
from torch.utils.data import Dataset
import torch
import random

class KAIST_Dataset(Dataset):
    def __init__(self, root_dir, scale=4, transform=None, train=True, max_samples=None):
        """
        Args:
            root_dir: đường dẫn đến thư mục gốc (chứa train/ và val/)
            scale: scale factor (4)
            transform: transforms
            train: True cho train, False cho val
            max_samples: số lượng mẫu tối đa
        """
        self.transform = transform
        self.scale = scale
        self.max_samples = max_samples
        
        # Kích thước patch cố định
        self.gt_h, self.gt_w = 128, 160  # GT patch: 128x160
        self.lr_h, self.lr_w = self.gt_h // scale, self.gt_w // scale  # LR patch: 32x40
        
        # Kích thước ảnh gốc KAIST
        self.img_h, self.img_w = 512, 640  # KAIST cũng 640x512
        
        split = 'train' if train else 'val'
        data_dir = os.path.join(root_dir, split)
        
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.gt_dir = os.path.join(data_dir, 'gt')
        self.train = train
        
        # Kiểm tra thư mục
        print(f"\n=== Checking KAIST {split} directories ===")
        print(f"RGB dir: {self.rgb_dir} - exists: {os.path.exists(self.rgb_dir)}")
        print(f"Depth dir: {self.depth_dir} - exists: {os.path.exists(self.depth_dir)}")
        print(f"GT dir: {self.gt_dir} - exists: {os.path.exists(self.gt_dir)}")
        
        # Lấy danh sách file - KAIST dùng .jpg cho RGB, .png cho depth và gt
        rgb_files = glob(os.path.join(self.rgb_dir, '*.jpg')) + glob(os.path.join(self.rgb_dir, '*.png'))
        depth_files = glob(os.path.join(self.depth_dir, '*.png')) + glob(os.path.join(self.depth_dir, '*.jpg'))
        gt_files = glob(os.path.join(self.gt_dir, '*.png')) + glob(os.path.join(self.gt_dir, '*.jpg'))
        
        print(f"\nFound files:")
        print(f"  RGB: {len(rgb_files)} files")
        print(f"  Depth: {len(depth_files)} files")
        print(f"  GT: {len(gt_files)} files")
        
        # Tìm tên chung
        rgb_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in rgb_files}
        depth_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in depth_files}
        gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
        
        common_names = set(rgb_dict.keys()) & set(depth_dict.keys()) & set(gt_dict.keys())
        common_names = sorted(list(common_names))
        
        print(f"Common files: {len(common_names)}")
        if len(common_names) > 0:
            print(f"First 5: {common_names[:5]}")
        
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
        self.valid_tops = list(range(0, self.img_h - self.gt_h + 1, self.scale))
        self.valid_lefts = list(range(0, self.img_w - self.gt_w + 1, self.scale))
        
        print(f"\n=== KAIST {split} set ===")
        print(f"Total files: {len(common_names)}")
        print(f"Image size: {self.img_h}x{self.img_w}")
        print(f"GT patch size: {self.gt_h}x{self.gt_w}")
        print(f"LR patch size: {self.lr_h}x{self.lr_w}")
        print(f"Valid top positions: {len(self.valid_tops)}")
        print(f"Valid left positions: {len(self.valid_lefts)}")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Đọc ảnh
        rgb = cv2.imread(self.rgb_files[idx])
        if rgb is None:
            raise ValueError(f"Không thể đọc RGB file: {self.rgb_files[idx]}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(self.depth_files[idx], cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"Không thể đọc Depth file: {self.depth_files[idx]}")
        
        gt = cv2.imread(self.gt_files[idx], cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise ValueError(f"Không thể đọc GT file: {self.gt_files[idx]}")
        
        # Kiểm tra kích thước
        h, w = gt.shape
        if h != self.img_h or w != self.img_w:
            print(f"Warning: Ảnh có kích thước {h}x{w}, resize về {self.img_h}x{self.img_w}")
            rgb = cv2.resize(rgb, (self.img_w, self.img_h))
            depth = cv2.resize(depth, (self.img_w, self.img_h))
            gt = cv2.resize(gt, (self.img_w, self.img_h))
        
        # Chọn vị trí patch
        if self.train:
            top = random.choice(self.valid_tops)
            left = random.choice(self.valid_lefts)
        else:
            # Center crop cho validation
            top = self.valid_tops[len(self.valid_tops) // 2]
            left = self.valid_lefts[len(self.valid_lefts) // 2]
        
        # Cắt patch trên GT và RGB
        gt_patch = gt[top:top+self.gt_h, left:left+self.gt_w]
        rgb_patch = rgb[top:top+self.gt_h, left:left+self.gt_w, :]
        
        # Tạo ảnh LR và cắt patch
        depth_lr_full = cv2.resize(depth, (self.img_w // self.scale, self.img_h // self.scale))
        lr_top = top // self.scale
        lr_left = left // self.scale
        lr_patch = depth_lr_full[lr_top:lr_top+self.lr_h, lr_left:lr_left+self.lr_w]
        
        # Kiểm tra kích thước patch
        assert gt_patch.shape == (self.gt_h, self.gt_w), f"GT patch size sai: {gt_patch.shape}"
        assert rgb_patch.shape[:2] == (self.gt_h, self.gt_w), f"RGB patch size sai: {rgb_patch.shape}"
        assert lr_patch.shape == (self.lr_h, self.lr_w), f"LR patch size sai: {lr_patch.shape}"
        
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

        return {
            'guidance': rgb_patch,  # [3, 128, 160]
            'lr': lr_patch,         # [1, 32, 40]
            'gt': gt_patch           # [1, 128, 160]
        }