# data/vedai_dataloader.py
from torchvision import transforms
import numpy as np
import os
import cv2
from glob import glob
from torch.utils.data import Dataset
import torch
import random
import re

class VEDAI_Dataset(Dataset):
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
        self.gt_h, self.gt_w = 128, 128  # GT patch: 128x128
        self.lr_h, self.lr_w = self.gt_h // scale, self.gt_w // scale  # LR patch: 32x32
        
        # Kích thước ảnh gốc
        self.img_h, self.img_w = 512, 512  # Ảnh vuông 512x512
        
        split = 'train' if train else 'val'
        data_dir = os.path.join(root_dir, split)
        
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.gt_dir = os.path.join(data_dir, 'gt')
        self.train = train
        
        # Kiểm tra thư mục
        print(f"\n=== Checking VEDAI {split} directories ===")
        print(f"RGB dir: {self.rgb_dir} - exists: {os.path.exists(self.rgb_dir)}")
        print(f"Depth dir: {self.depth_dir} - exists: {os.path.exists(self.depth_dir)}")
        print(f"GT dir: {self.gt_dir} - exists: {os.path.exists(self.gt_dir)}")
        
        # Lấy danh sách file
        rgb_files = glob(os.path.join(self.rgb_dir, '*.png')) + glob(os.path.join(self.rgb_dir, '*.jpg'))
        depth_files = glob(os.path.join(self.depth_dir, '*.png')) + glob(os.path.join(self.depth_dir, '*.jpg'))
        gt_files = glob(os.path.join(self.gt_dir, '*.png')) + glob(os.path.join(self.gt_dir, '*.jpg'))
        
        print(f"\nFound files:")
        print(f"  RGB: {len(rgb_files)} files")
        print(f"  Depth: {len(depth_files)} files")
        print(f"  GT: {len(gt_files)} files")
        
        # Hàm trích xuất số 8 chữ số từ tên file
        def extract_number(filename):
            # Lấy tên file không có path
            basename = os.path.basename(filename)
            # Tìm 8 số đầu tiên
            match = re.match(r'(\d{8})', basename)
            if match:
                return match.group(1)
            else:
                # Nếu không tìm thấy, trả về tên file không extension
                return os.path.splitext(basename)[0]
        
        # Tạo dictionary với key là 8 số đầu
        rgb_dict = {}
        for f in rgb_files:
            key = extract_number(f)
            if key not in rgb_dict:  # Giữ file đầu tiên nếu trùng
                rgb_dict[key] = f
        
        depth_dict = {}
        for f in depth_files:
            key = extract_number(f)
            if key not in depth_dict:
                depth_dict[key] = f
        
        gt_dict = {}
        for f in gt_files:
            key = extract_number(f)
            if key not in gt_dict:
                gt_dict[key] = f
        
        # Tìm tên chung (8 số đầu)
        common_numbers = set(rgb_dict.keys()) & set(depth_dict.keys()) & set(gt_dict.keys())
        common_numbers = sorted(list(common_numbers))
        
        print(f"\nCommon 8-digit numbers: {len(common_numbers)}")
        if len(common_numbers) > 0:
            print(f"First 5 numbers: {common_numbers[:5]}")
        
        # Lấy đường dẫn file tương ứng
        self.rgb_files = [rgb_dict[num] for num in common_numbers]
        self.depth_files = [depth_dict[num] for num in common_numbers]
        self.gt_files = [gt_dict[num] for num in common_numbers]
        
        # Giới hạn số lượng samples
        if max_samples is not None and len(self.rgb_files) > max_samples:
            if train:
                # Random sample
                indices = random.sample(range(len(self.rgb_files)), max_samples)
                self.rgb_files = [self.rgb_files[i] for i in indices]
                self.depth_files = [self.depth_files[i] for i in indices]
                self.gt_files = [self.gt_files[i] for i in indices]
            else:
                # Lấy max_samples đầu tiên
                self.rgb_files = self.rgb_files[:max_samples]
                self.depth_files = self.depth_files[:max_samples]
                self.gt_files = self.gt_files[:max_samples]
        
        # Tính các vị trí top, left hợp lệ (chia hết cho scale)
        self.valid_tops = list(range(0, self.img_h - self.gt_h + 1, self.scale))
        self.valid_lefts = list(range(0, self.img_w - self.gt_w + 1, self.scale))
        
        print(f"\n=== VEDAI {split} set ===")
        print(f"Total valid files: {len(self.rgb_files)}")
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
        
        # Depth có kích thước 128x128 (LR) - kiểm tra
        h_depth, w_depth = depth.shape
        if h_depth != self.img_h // self.scale or w_depth != self.img_w // self.scale:
            print(f"Warning: Depth có kích thước {h_depth}x{w_depth}, resize về {self.img_h//self.scale}x{self.img_w//self.scale}")
            depth = cv2.resize(depth, (self.img_w // self.scale, self.img_h // self.scale))
        
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
        
        # Tính vị trí tương ứng trên depth
        lr_top = top // self.scale
        lr_left = left // self.scale
        lr_patch = depth[lr_top:lr_top+self.lr_h, lr_left:lr_left+self.lr_w]
        
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
            'guidance': rgb_patch,  # [3, 128, 128]
            'lr': lr_patch,         # [1, 32, 32]
            'gt': gt_patch           # [1, 128, 128]
        }