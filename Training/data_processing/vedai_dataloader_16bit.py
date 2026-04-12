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
    # Biến class để lưu homography từ train set
    train_H = None
    
    def __init__(self, root_dir, scale=4, transform=None, train=True, max_samples=None, use_homography=True):
        """
        Args:
            root_dir: đường dẫn đến thư mục gốc (chứa train/ và val/)
            scale: scale factor (4)
            transform: transforms
            train: True cho train, False cho val
            max_samples: số lượng mẫu tối đa
            use_homography: True để tự động tính homography, False để chỉ resize
        """
        self.transform = transform
        self.scale = scale
        self.max_samples = max_samples
        self.use_homography = use_homography
        self.train = train
        
        # Kích thước patch cố định
        self.gt_h, self.gt_w = 128, 128  # GT patch: 128x128
        self.lr_h, self.lr_w = self.gt_h // scale, self.gt_w // scale  # LR patch: 32x32
        
        split = 'train' if train else 'val'
        data_dir = os.path.join(root_dir, split)
        
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.gt_dir = os.path.join(data_dir, 'gt')
        
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
            basename = os.path.basename(filename)
            match = re.match(r'(\d{8})', basename)
            return match.group(1) if match else os.path.splitext(basename)[0]
        
        # Tạo dictionary với key là 8 số đầu
        rgb_dict = {}
        for f in rgb_files:
            key = extract_number(f)
            if key not in rgb_dict:
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
        
        # Tìm tên chung
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
                indices = random.sample(range(len(self.rgb_files)), max_samples)
                self.rgb_files = [self.rgb_files[i] for i in indices]
                self.depth_files = [self.depth_files[i] for i in indices]
                self.gt_files = [self.gt_files[i] for i in indices]
            else:
                self.rgb_files = self.rgb_files[:max_samples]
                self.depth_files = self.depth_files[:max_samples]
                self.gt_files = self.gt_files[:max_samples]
        
        # XỬ LÝ HOMOGRAPHY
        self.H = np.eye(3)
        
        if use_homography and len(self.rgb_files) > 0:
            if train:
                self.H = self.compute_homography_from_first_pair()
                VEDAI_Dataset.train_H = self.H
                print(f"\n✅ Homography matrix computed for TRAIN:\n{self.H}")
            else:
                if VEDAI_Dataset.train_H is not None:
                    self.H = VEDAI_Dataset.train_H
                    print(f"\n✅ Using homography from TRAIN for VAL set")
                else:
                    print(f"\n⚠️ No train homography available, using identity for VAL")
        
        print(f"\n=== VEDAI {split} set ===")
        print(f"Total valid files: {len(self.rgb_files)}")
        print(f"Using homography: {use_homography}")

    def __len__(self):
        return len(self.rgb_files)

    def compute_homography_from_first_pair(self):
        """Tính homography từ cặp ảnh đầu tiên"""
        try:
            rgb = cv2.imread(self.rgb_files[0])
            if rgb is None:
                return np.eye(3)
            rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            
            depth = cv2.imread(self.depth_files[0], cv2.IMREAD_GRAYSCALE)
            if depth is None:
                return np.eye(3)
            
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(rgb_gray, None)
            kp2, des2 = sift.detectAndCompute(depth, None)
            
            if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
                return np.eye(3)
            
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 4:
                return np.eye(3)
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H if H is not None else np.eye(3)
            
        except Exception as e:
            return np.eye(3)

    def apply_homography_and_resize(self, rgb, target_size):
        """Áp dụng homography để warp RGB"""
        if self.use_homography and self.H is not None:
            rgb_warped = cv2.warpPerspective(
                rgb, self.H, target_size,
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            rgb_warped = cv2.resize(rgb, target_size, interpolation=cv2.INTER_CUBIC)
        return rgb_warped

    def __getitem__(self, idx):
        # Đọc ảnh RGB
        rgb = cv2.imread(self.rgb_files[idx])
        if rgb is None:
            raise ValueError(f"Không thể đọc RGB file: {self.rgb_files[idx]}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Đọc depth
        depth = cv2.imread(self.depth_files[idx], cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"Không thể đọc Depth file: {self.depth_files[idx]}")
        
        # Đọc GT
        gt = cv2.imread(self.gt_files[idx], cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise ValueError(f"Không thể đọc GT file: {self.gt_files[idx]}")
        
        # Lấy kích thước thực tế của ảnh
        h, w = gt.shape
        
        # Đảm bảo depth và RGB cùng kích thước với GT
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h))
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h))
        
        # Áp dụng homography
        rgb_processed = self.apply_homography_and_resize(rgb, (w, h))
        
        # Tính các vị trí top, left hợp lệ dựa trên kích thước thực tế
        valid_tops = list(range(0, h - self.gt_h + 1, self.scale))
        valid_lefts = list(range(0, w - self.gt_w + 1, self.scale))
        
        if len(valid_tops) == 0 or len(valid_lefts) == 0:
            # Nếu ảnh quá nhỏ, resize lên
            new_h, new_w = self.gt_h * 2, self.gt_w * 2
            gt = cv2.resize(gt, (new_w, new_h))
            rgb_processed = cv2.resize(rgb_processed, (new_w, new_h))
            depth = cv2.resize(depth, (new_w, new_h))
            h, w = new_h, new_w
            valid_tops = list(range(0, h - self.gt_h + 1, self.scale))
            valid_lefts = list(range(0, w - self.gt_w + 1, self.scale))
        
        # Tạo ảnh LR
        depth_lr_full = cv2.resize(depth, (w // self.scale, h // self.scale))
        
        # Chọn vị trí patch
        if self.train:
            top = random.choice(valid_tops)
            left = random.choice(valid_lefts)
        else:
            top = valid_tops[len(valid_tops) // 2] if valid_tops else 0
            left = valid_lefts[len(valid_lefts) // 2] if valid_lefts else 0
        
        # Cắt patch
        gt_patch = gt[top:top+self.gt_h, left:left+self.gt_w]
        rgb_patch = rgb_processed[top:top+self.gt_h, left:left+self.gt_w, :]
        
        lr_top = top // self.scale
        lr_left = left // self.scale
        lr_patch = depth_lr_full[lr_top:lr_top+self.lr_h, lr_left:lr_left+self.lr_w]
        
        # Resize nếu cần
        if gt_patch.shape != (self.gt_h, self.gt_w):
            gt_patch = cv2.resize(gt_patch, (self.gt_w, self.gt_h))
        if rgb_patch.shape[:2] != (self.gt_h, self.gt_w):
            rgb_patch = cv2.resize(rgb_patch, (self.gt_w, self.gt_h))
        if lr_patch.shape != (self.lr_h, self.lr_w):
            lr_patch = cv2.resize(lr_patch, (self.lr_w, self.lr_h))
        
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
            'guidance': rgb_patch,
            'lr': lr_patch,
            'gt': gt_patch
        }