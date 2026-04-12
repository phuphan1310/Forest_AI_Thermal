# data/flame3_dataloader.py
from torchvision import transforms
import numpy as np
import os
import cv2
from glob import glob
from torch.utils.data import Dataset
import torch
import random
import re

class Flame3_Dataset(Dataset):
    # Biến class để lưu homography từ train set
    train_H = None
    
    def __init__(self, root_dir, scale=4, transform=None, train=True, max_samples=1000, use_homography=True):
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
        
        # Kích thước patch cố định (giống FLIR)
        self.gt_h, self.gt_w = 128, 160  # GT patch: 128x160
        self.lr_h, self.lr_w = self.gt_h // scale, self.gt_w // scale  # LR patch: 32x40
        
        # Kích thước các ảnh
        self.rgb_orig_h, self.rgb_orig_w = 6000, 8000  # RGB gốc: 6000x8000
        self.depth_h, self.depth_w = 128, 160          # Depth gốc: 128x160
        self.gt_h_orig, self.gt_w_orig = 512, 640      # GT gốc (giả định, có thể điều chỉnh)
        
        split = 'train' if train else 'val'
        data_dir = os.path.join(root_dir, split)
        
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.gt_dir = os.path.join(data_dir, 'gt')
        
        # Lấy danh sách file
        rgb_files = glob(os.path.join(self.rgb_dir, '*.jpg')) + glob(os.path.join(self.rgb_dir, '*.png')) + glob(os.path.join(self.rgb_dir, '*.tiff'))
        depth_files = glob(os.path.join(self.depth_dir, '*.tiff')) + glob(os.path.join(self.depth_dir, '*.tif')) + glob(os.path.join(self.depth_dir, '*.png'))
        gt_files = glob(os.path.join(self.gt_dir, '*.tiff')) + glob(os.path.join(self.gt_dir, '*.tif')) + glob(os.path.join(self.gt_dir, '*.png'))
        
        print(f"\nFound files in {split}:")
        print(f"  RGB (8000x6000): {len(rgb_files)} files")
        print(f"  Depth (160x128): {len(depth_files)} files")
        print(f"  GT (640x512): {len(gt_files)} files")
        
        # Trích xuất số từ tên file
        def extract_number(filename):
            basename = os.path.basename(filename)
            match = re.search(r'(\d+)', basename)
            return match.group(1) if match else os.path.splitext(basename)[0]
        
        rgb_dict = {extract_number(f): f for f in rgb_files}
        depth_dict = {extract_number(f): f for f in depth_files}
        gt_dict = {extract_number(f): f for f in gt_files}
        
        common_names = set(rgb_dict.keys()) & set(depth_dict.keys()) & set(gt_dict.keys())
        common_names = sorted(list(common_names))
        
        print(f"Common files: {len(common_names)}")
        
        # Giới hạn số lượng samples
        if len(common_names) > max_samples:
            if train:
                common_names = random.sample(common_names, max_samples)
            else:
                common_names = common_names[:max_samples]
        
        self.rgb_files = [rgb_dict[name] for name in common_names]
        self.depth_files = [depth_dict[name] for name in common_names]
        self.gt_files = [gt_dict[name] for name in common_names]
        
        # XỬ LÝ HOMOGRAPHY
        self.H = np.eye(3)  # Mặc định là identity
        
        if use_homography and len(self.rgb_files) > 0:
            if train:
                # Nếu là train, tính homography mới
                self.H = self.compute_homography_from_first_pair()
                Flame3_Dataset.train_H = self.H  # Lưu lại để dùng cho val
                print(f"\n✅ Homography matrix computed for TRAIN:\n{self.H}")
            else:
                # Nếu là val, dùng homography từ train
                if Flame3_Dataset.train_H is not None:
                    self.H = Flame3_Dataset.train_H
                    print(f"\n✅ Using homography from TRAIN for VAL set")
                else:
                    print(f"\n⚠️ No train homography available, using identity for VAL")
        
        # Tính các vị trí top, left hợp lệ cho GT
        self.valid_tops = list(range(0, self.gt_h_orig - self.gt_h + 1, self.scale))
        self.valid_lefts = list(range(0, self.gt_w_orig - self.gt_w + 1, self.scale))
        
        print(f"\n=== {split} set ===")
        print(f"Total files: {len(self.rgb_files)}")
        print(f"Using homography: {use_homography}")
        print(f"GT patch size: {self.gt_h}x{self.gt_w}")
        print(f"LR patch size: {self.lr_h}x{self.lr_w}")
        print(f"GT original size: {self.gt_h_orig}x{self.gt_w_orig}")
        print(f"Valid top range: 0 to {self.gt_h_orig - self.gt_h}")
        print(f"Valid left range: 0 to {self.gt_w_orig - self.gt_w}")

    def __len__(self):
        """Trả về số lượng samples trong dataset"""
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
            
            depth = cv2.imread(self.depth_files[0], cv2.IMREAD_UNCHANGED)
            if depth is None:
                return np.eye(3)
            
            # Chuẩn hóa depth để dùng SIFT
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Resize depth lên kích thước lớn hơn để dễ tìm feature
            depth_resized = cv2.resize(depth_norm, (640, 512))
            
            # Tạo feature detector
            sift = cv2.SIFT_create()
            
            # Tìm keypoints và descriptors
            kp1, des1 = sift.detectAndCompute(rgb_gray, None)
            kp2, des2 = sift.detectAndCompute(depth_resized, None)
            
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

    def apply_homography_and_resize(self, rgb):
        """
        Áp dụng homography và resize để đưa RGB về cùng kích thước với GT
        """
        if self.use_homography and self.H is not None:
            # Warp RGB về cùng góc nhìn với depth
            rgb_warped = cv2.warpPerspective(
                rgb, 
                self.H, 
                (self.depth_w, self.depth_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            # Nếu không dùng homography, chỉ resize thẳng xuống depth
            rgb_warped = cv2.resize(rgb, (self.depth_w, self.depth_h), interpolation=cv2.INTER_CUBIC)
        
        # Resize lên kích thước GT
        rgb_final = cv2.resize(
            rgb_warped, 
            (self.gt_w_orig, self.gt_h_orig), 
            interpolation=cv2.INTER_CUBIC
        )
        
        return rgb_final

    def __getitem__(self, idx):
        # Đọc ảnh RGB (8000x6000)
        rgb = cv2.imread(self.rgb_files[idx])
        if rgb is None:
            # Nếu lỗi, lấy ảnh đầu tiên
            rgb = cv2.imread(self.rgb_files[0])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Đọc depth (160x128) - có thể 16-bit
        depth = cv2.imread(self.depth_files[idx], cv2.IMREAD_UNCHANGED)
        if depth is None:
            depth = cv2.imread(self.depth_files[0], cv2.IMREAD_UNCHANGED)
        
        # Đọc GT (640x512) - có thể 16-bit
        gt = cv2.imread(self.gt_files[idx], cv2.IMREAD_UNCHANGED)
        if gt is None:
            gt = cv2.imread(self.gt_files[0], cv2.IMREAD_UNCHANGED)
        
        # Áp dụng homography và resize cho RGB
        rgb_processed = self.apply_homography_and_resize(rgb)
        
        # Resize depth lên kích thước GT
        depth_resized = cv2.resize(depth, (self.gt_w_orig, self.gt_h_orig), interpolation=cv2.INTER_CUBIC)
        
        # Chọn vị trí patch trên GT
        if self.train:
            # Random crop cho training
            top = random.randint(0, max(0, gt.shape[0] - self.gt_h))
            left = random.randint(0, max(0, gt.shape[1] - self.gt_w))
        else:
            # Center crop cho validation
            top = max(0, (gt.shape[0] - self.gt_h) // 2)
            left = max(0, (gt.shape[1] - self.gt_w) // 2)
        
        # Đảm bảo patch nằm trong ảnh
        top = min(top, max(0, gt.shape[0] - self.gt_h))
        left = min(left, max(0, gt.shape[1] - self.gt_w))
        
        # Cắt patch trên GT
        gt_patch = gt[top:top+self.gt_h, left:left+self.gt_w]
        
        # Cắt patch trên RGB đã xử lý (cùng vị trí)
        rgb_patch = rgb_processed[top:top+self.gt_h, left:left+self.gt_w, :]
        
        # Cắt patch trên depth đã resize
        depth_patch = depth_resized[top:top+self.gt_h, left:left+self.gt_w]
        
        # Kiểm tra và resize nếu kích thước không đúng
        if gt_patch.shape[0] != self.gt_h or gt_patch.shape[1] != self.gt_w:
            gt_patch = cv2.resize(gt_patch, (self.gt_w, self.gt_h))
        
        if rgb_patch.shape[0] != self.gt_h or rgb_patch.shape[1] != self.gt_w:
            rgb_patch = cv2.resize(rgb_patch, (self.gt_w, self.gt_h))
        
        if depth_patch.shape[0] != self.gt_h or depth_patch.shape[1] != self.gt_w:
            depth_patch = cv2.resize(depth_patch, (self.gt_w, self.gt_h))
        
        # Tạo LR patch bằng cách downsample
        lr_patch = cv2.resize(depth_patch, (self.lr_w, self.lr_h), interpolation=cv2.INTER_CUBIC)
        
        # Xác định bit depth để normalize
        if rgb_patch.dtype == np.uint16:
            rgb_patch = (rgb_patch / 65535.0).astype(np.float32)
        else:
            rgb_patch = rgb_patch.astype(np.float32) / 255.0
            
        if lr_patch.dtype == np.uint16:
            lr_patch = lr_patch.astype(np.float32) / 65535.0
        else:
            lr_patch = lr_patch.astype(np.float32) / 255.0
            
        if gt_patch.dtype == np.uint16:
            gt_patch = gt_patch.astype(np.float32) / 65535.0
        else:
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