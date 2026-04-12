# data_processing/kaist_dataloader_16bit.py
from torchvision import transforms
import numpy as np
import os
import cv2
from glob import glob
from torch.utils.data import Dataset
import torch
import random
import re

class KAIST_Dataset(Dataset):
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
        self.gt_h, self.gt_w = 128, 160  # GT patch: 128x160
        self.lr_h, self.lr_w = self.gt_h // scale, self.gt_w // scale  # LR patch: 32x40
        
        # Kích thước ảnh gốc KAIST
        self.img_h, self.img_w = 512, 640  # KAIST 640x512
        
        split = 'train' if train else 'val'
        data_dir = os.path.join(root_dir, split)
        
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.gt_dir = os.path.join(data_dir, 'gt')
        
        # Kiểm tra thư mục
        print(f"\n=== Checking KAIST {split} directories ===")
        print(f"RGB dir: {self.rgb_dir} - exists: {os.path.exists(self.rgb_dir)}")
        print(f"Depth dir: {self.depth_dir} - exists: {os.path.exists(self.depth_dir)}")
        print(f"GT dir: {self.gt_dir} - exists: {os.path.exists(self.gt_dir)}")
        
        # Lấy danh sách file
        rgb_files = glob(os.path.join(self.rgb_dir, '*.jpg')) + glob(os.path.join(self.rgb_dir, '*.png'))
        depth_files = glob(os.path.join(self.depth_dir, '*.png')) + glob(os.path.join(self.depth_dir, '*.jpg'))
        gt_files = glob(os.path.join(self.gt_dir, '*.png')) + glob(os.path.join(self.gt_dir, '*.jpg'))
        
        print(f"\nFound files:")
        print(f"  RGB: {len(rgb_files)} files")
        print(f"  Depth: {len(depth_files)} files")
        print(f"  GT: {len(gt_files)} files")
        
        # In mẫu tên file để kiểm tra
        if len(rgb_files) > 0:
            print(f"\nSample RGB filenames: {[os.path.basename(f) for f in rgb_files[:3]]}")
            print(f"Sample Depth filenames: {[os.path.basename(f) for f in depth_files[:3]]}")
            print(f"Sample GT filenames: {[os.path.basename(f) for f in gt_files[:3]]}")
        
        # SỬA: Dùng cả tên file không extension làm key
        rgb_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in rgb_files}
        depth_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in depth_files}
        gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
        
        common_names = set(rgb_dict.keys()) & set(depth_dict.keys()) & set(gt_dict.keys())
        common_names = sorted(list(common_names))
        
        print(f"Common files: {len(common_names)}")
        if len(common_names) > 0:
            print(f"First 5 common names: {common_names[:5]}")
        
        # Giới hạn số lượng samples
        if max_samples is not None and len(common_names) > max_samples:
            if train:
                common_names = random.sample(common_names, max_samples)
            else:
                common_names = common_names[:max_samples]
        
        self.rgb_files = [rgb_dict[name] for name in common_names]
        self.depth_files = [depth_dict[name] for name in common_names]
        self.gt_files = [gt_dict[name] for name in common_names]
        
        print(f"\n=== Final dataset size: {len(self.rgb_files)} samples ===")
        
        # XỬ LÝ HOMOGRAPHY
        self.H = np.eye(3)  # Mặc định là identity
        
        if use_homography and len(self.rgb_files) > 0:
            if train:
                # Nếu là train, tính homography mới
                self.H = self.compute_homography_from_first_pair()
                KAIST_Dataset.train_H = self.H  # Lưu lại để dùng cho val
                print(f"\n✅ Homography matrix computed for TRAIN:\n{self.H}")
            else:
                # Nếu là val, dùng homography từ train
                if KAIST_Dataset.train_H is not None:
                    self.H = KAIST_Dataset.train_H
                    print(f"\n✅ Using homography from TRAIN for VAL set")
                else:
                    print(f"\n⚠️ No train homography available, using identity for VAL")
        
        # Tính các vị trí top, left hợp lệ (chia hết cho scale)
        self.valid_tops = list(range(0, self.img_h - self.gt_h + 1, self.scale))
        self.valid_lefts = list(range(0, self.img_w - self.gt_w + 1, self.scale))
        
        print(f"\n=== KAIST {split} set ===")
        print(f"Total files: {len(self.rgb_files)}")
        print(f"Using homography: {use_homography}")
        print(f"Image size: {self.img_h}x{self.img_w}")
        print(f"GT patch size: {self.gt_h}x{self.gt_w}")
        print(f"LR patch size: {self.lr_h}x{self.lr_w}")
        print(f"Valid top positions: {len(self.valid_tops)}")
        print(f"Valid left positions: {len(self.valid_lefts)}")

    def __len__(self):
        return len(self.rgb_files)

    def compute_homography_from_first_pair(self):
        """
        Tự động tính homography từ cặp RGB và depth đầu tiên
        Dùng feature matching (SIFT) để tìm điểm tương đồng
        """
        try:
            # Đọc ảnh đầu tiên
            rgb = cv2.imread(self.rgb_files[0])
            if rgb is None:
                return np.eye(3)
            rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            
            depth = cv2.imread(self.depth_files[0], cv2.IMREAD_GRAYSCALE)
            if depth is None:
                return np.eye(3)
            
            # Tạo feature detector
            sift = cv2.SIFT_create()
            
            # Tìm keypoints và descriptors
            kp1, des1 = sift.detectAndCompute(rgb_gray, None)
            kp2, des2 = sift.detectAndCompute(depth, None)
            
            if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
                print("⚠️ Không đủ features, dùng identity matrix")
                return np.eye(3)
            
            # Matching features
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Lọc matches tốt
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            print(f"Found {len(good_matches)} good matches")
            
            if len(good_matches) < 4:
                print("⚠️ Không đủ good matches, dùng identity matrix")
                return np.eye(3)
            
            # Lấy tọa độ các điểm match
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Tính homography
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            return H if H is not None else np.eye(3)
            
        except Exception as e:
            print(f"⚠️ Lỗi khi tính homography: {e}")
            return np.eye(3)

    def apply_homography_and_resize(self, rgb, target_size):
        """
        Áp dụng homography để warp RGB về cùng góc nhìn với depth
        """
        if self.use_homography and self.H is not None:
            rgb_warped = cv2.warpPerspective(
                rgb, 
                self.H, 
                target_size,
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
        
        # Kiểm tra kích thước
        h, w = gt.shape
        if h != self.img_h or w != self.img_w:
            print(f"Warning: Ảnh có kích thước {h}x{w}, resize về {self.img_h}x{self.img_w}")
            gt = cv2.resize(gt, (self.img_w, self.img_h))
            depth = cv2.resize(depth, (self.img_w, self.img_h))
        
        # Áp dụng homography để warp RGB về cùng kích thước với GT
        rgb_processed = self.apply_homography_and_resize(rgb, (self.img_w, self.img_h))
        
        # Tạo ảnh LR từ depth
        depth_lr_full = cv2.resize(depth, (self.img_w // self.scale, self.img_h // self.scale), interpolation=cv2.INTER_CUBIC)
        
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
        rgb_patch = rgb_processed[top:top+self.gt_h, left:left+self.gt_w, :]
        
        # Tính vị trí trên LR
        lr_top = top // self.scale
        lr_left = left // self.scale
        lr_patch = depth_lr_full[lr_top:lr_top+self.lr_h, lr_left:lr_left+self.lr_w]
        
        # Kiểm tra kích thước patch
        assert gt_patch.shape == (self.gt_h, self.gt_w), f"GT patch size sai: {gt_patch.shape}"
        assert rgb_patch.shape[:2] == (self.gt_h, self.gt_w), f"RGB patch size sai: {rgb_patch.shape}"
        assert lr_patch.shape == (self.lr_h, self.lr_w), f"LR patch size sai: {lr_patch.shape}"
        
        # Normalize (8-bit)
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