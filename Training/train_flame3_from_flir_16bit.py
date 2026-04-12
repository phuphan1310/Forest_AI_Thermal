# train_flame3_from_flir.py
import os
import torch
import numpy as np
import argparse
from models.SGNet import SGNet
from models.common import *
from data_processing.flame3_dataloader import Flame3_Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import logging
from datetime import datetime
import multiprocessing as mp
import torch.backends.cudnn as cudnn

# Fix cuDNN
cudnn.benchmark = False
cudnn.deterministic = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument('--lr', default='0.00005', type=float, help='learning rate (nhỏ hơn cho fine-tune)')
parser.add_argument('--result', default='experiment', help='result folder')
parser.add_argument('--epoch', default=100, type=int, help='max epoch (ít hơn cho fine-tune)')
parser.add_argument('--device', default="0", type=str, help='which gpu use')
parser.add_argument("--decay_iterations", type=list, default=[2e4, 4e4, 6e4], help="steps to start lr decay")
parser.add_argument("--num_feats", type=int, default=32, help="channel number of the middle hidden layer")
parser.add_argument("--gamma", type=float, default=0.2, help="decay rate of learning rate")
parser.add_argument("--root_dir", type=str, default='C:/Users/ADMIN/MySGNet/data/flame3', 
                    help="root dir of Flame3 dataset")
parser.add_argument("--batchsize", type=int, default=2, help="batchsize of training dataloader (giảm vì ảnh lớn)")
parser.add_argument("--max_samples", type=int, default=1000, help="max samples for training")
parser.add_argument("--use_homography", type=bool, default=True, help="use homography for alignment")
parser.add_argument("--pretrained", type=str, 
                    default='C:/Users/ADMIN/Guided SR in thermal images/experiments_flir_16bit/20260228193500-lr_0.0001-s_4-flir_data-b_4/best_model_rmse_0.0004_epoch94.pth',
                    help="path to pretrained FLIR model")

if __name__ == '__main__':
    mp.freeze_support()
    
    # Clear cache
    torch.cuda.empty_cache()
    
    opt = parser.parse_args()
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    s = datetime.now().strftime('%Y%m%d%H%M%S')
    result_root = '%s/Flame3-finetune-from-flir-lr_%s-s_%s-b_%s-%s' % (opt.result, opt.lr, opt.scale, opt.batchsize, s)
    if not os.path.exists(result_root):
        os.makedirs(result_root)

    logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info(opt)

    # Khởi tạo model
    net = SGNet(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale).cuda()
    
    # Load pretrained weights từ FLIR
    print(f"\nLoading pretrained FLIR model from: {opt.pretrained}")
    print("  → Model đã được train trên FLIR 16bit với RMSE: 0.0004")
    checkpoint = torch.load(opt.pretrained)
    net.load_state_dict(checkpoint)
    print("✓ Pretrained weights loaded successfully!")
    
    net_getFre = get_Fre()
    net_grad = Get_gradient_nopadding_d()

    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.decay_iterations, gamma=opt.gamma)
    net.train()

    data_transform = transforms.Compose([transforms.ToTensor()])

    # Load Flame3 dataset
    print("\nLoading Flame3 dataset for fine-tuning...")
    train_dataset = Flame3_Dataset(
        root_dir=opt.root_dir,
        scale=opt.scale,
        transform=data_transform,
        train=True,
        max_samples=opt.max_samples,
        use_homography=opt.use_homography
    )
    val_dataset = Flame3_Dataset(
        root_dir=opt.root_dir,
        scale=opt.scale,
        transform=data_transform,
        train=False,
        max_samples=200,
        use_homography=opt.use_homography
    )

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=opt.batchsize, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    max_epoch = opt.epoch
    num_train = len(train_dataloader)
    best_rmse = 100.0
    best_epoch = 0

    print(f"\n=== Starting Flame3 Fine-tuning from FLIR ===")
    print(f"Train batches: {num_train}")
    print(f"Val batches: {len(val_dataloader)}")
    print(f"Total epochs: {max_epoch}")
    print(f"Using homography: {opt.use_homography}")

    for epoch in range(max_epoch):
        # Training
        net.train()
        running_loss = 0.0

        t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))

        for idx, data in enumerate(t):
            try:
                batches_done = num_train * epoch + idx
                
                guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()
                
                # Debug shapes lần đầu
                if idx == 0 and epoch == 0:
                    print(f"\nGuidance shape: {guidance.shape}")
                    print(f"LR shape: {lr.shape}")
                    print(f"GT shape: {gt.shape}\n")

                out, out_grad = net((guidance, lr))

                out_amp, out_pha = net_getFre(out)
                gt_amp, gt_pha = net_getFre(gt)

                gt_grad = net_grad(gt)
                loss_grad1 = criterion(out_grad, gt_grad)

                loss_fre_amp = criterion(out_amp, gt_amp)
                loss_fre_pha = criterion(out_pha, gt_pha)

                loss_fre = 0.5 * loss_fre_amp + 0.5 * loss_fre_pha

                loss_spa = criterion(out, gt)

                loss = loss_spa + 0.002 * loss_fre + 0.001 * loss_grad1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.data.item()

                if idx % 50 == 0:
                    t.set_description('[Flame3 finetune epoch:%d] loss: %.6f' % (epoch + 1, loss.data.item()))
                    
            except RuntimeError as e:
                print(f"Error at batch {idx}: {e}")
                torch.cuda.empty_cache()
                continue

        scheduler.step()
        
        avg_loss = running_loss / len(train_dataloader)
        logging.info('epoch:%d loss:%.6f lr:%.6f' % (epoch + 1, avg_loss, scheduler.get_last_lr()[0]))

        # Validation mỗi 2 epochs
        if epoch % 2 == 0:
            with torch.no_grad():
                net.eval()
                
                rmse = np.zeros(len(val_dataloader))
                
                t = tqdm(iter(val_dataloader), leave=True, total=len(val_dataloader))

                for idx, data in enumerate(t):
                    guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()
                    out, out_grad = net((guidance, lr))
                    
                    # Tính RMSE
                    mse = torch.mean((gt[0, 0] - out[0, 0]) ** 2)
                    rmse[idx] = torch.sqrt(mse).cpu().numpy()

                    t.set_description('[Flame3 validate] rmse: %.4f' % rmse[:idx + 1].mean())

                r_mean = rmse.mean()
                
                # Lưu best model
                if r_mean < best_rmse:
                    best_rmse = r_mean
                    best_epoch = epoch
                    torch.save(net.state_dict(),
                              os.path.join(result_root, "best_model_rmse_%.4f_epoch%d.pth" % (best_rmse, best_epoch)))
                    print(f"\n✨ New best model! RMSE: {best_rmse:.4f} at epoch {best_epoch+1}")
                
                logging.info('-'*60)
                logging.info('epoch:%d lr:%f - mean_rmse:%.4f (BEST: %.4f @epoch%d)' % (
                    epoch + 1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch + 1))
                logging.info('-'*60)
        
        # Lưu checkpoint mỗi 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(net.state_dict(),
                      os.path.join(result_root, "checkpoint_epoch%d.pth" % (epoch + 1)))