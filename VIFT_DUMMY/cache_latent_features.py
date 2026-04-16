"""
cache_latent_features.py
========================
Stage 1 of the VIFT pipeline: Use the ORIGINAL pre-trained encoder to cache
768-dim latent features (visual 512 + inertial 256) as .npy files.

The pre-trained weights (vf_512_if_256_3e-05.model) were trained with the
ORIGINAL heavy encoder architecture from the parent VIFT project.
We import that original architecture here for weight compatibility.

Usage:
    python cache_latent_features.py --split train
    python cache_latent_features.py --split val
"""

import sys
import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

# Import KITTI dataset and transforms from VIFT_DUMMY
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from KITTI_dataset import KITTI
import custom_transform

# ─────────────────────────────────────────────────────────────────
# ORIGINAL Encoder Architecture (from parent VIFT project)
# Must match the architecture that produced vf_512_if_256_3e-05.model
# ─────────────────────────────────────────────────────────────────

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0.0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

class OriginalInertialEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))
        x = self.encoder_conv(x.permute(0, 2, 1))
        out = self.proj(x.view(x.shape[0], -1))
        return out.view(batch_size, seq_len, 256)


class OriginalEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)

        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)
        self.inertial_encoder = OriginalInertialEncoder(opt)

    def forward(self, img, imu):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        v = v.view(batch_size, seq_len, -1)
        v = self.visual_head(v)

        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)
        return v, imu

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


class FeatureEncodingModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Feature_net = OriginalEncoder(params)
    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        return feat_v, feat_i


def main():
    parser = argparse.ArgumentParser(description='Cache latent features for transformer training')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Which split to cache (train or val)')
    parser.add_argument('--data_dir', type=str, default='data/kitti_data',
                        help='Path to KITTI data directory')
    parser.add_argument('--weights', type=str, default='pretrained_models/vf_512_if_256_3e-05.model',
                        help='Path to pre-trained encoder weights')
    parser.add_argument('--seq_len', type=int, default=11,
                        help='Sequence length')
    args = parser.parse_args()

    # Define train/val sequences (same as VIFT)
    if args.split == 'train':
        seqs = ['00', '01', '02', '04', '06', '08', '09']
        save_dir = os.path.join('data', 'kitti_latent_data', f'train_{args.seq_len}')
    else:
        seqs = ['05', '07', '10']
        save_dir = os.path.join('data', 'kitti_latent_data', f'val_{args.seq_len}')

    print(f"[INFO] Caching {args.split} split: sequences {seqs}")
    print(f"[INFO] Save directory: {save_dir}")

    # Setup transforms
    transform = custom_transform.Compose([
        custom_transform.ToTensor(),
        custom_transform.Resize((256, 512))
    ])

    # Load dataset
    dataset = KITTI(args.data_dir, train_seqs=seqs, transform=transform, sequence_length=args.seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")

    # Create model with ORIGINAL architecture
    from types import SimpleNamespace
    params = SimpleNamespace(
        img_w=512, img_h=256,
        v_f_len=512, i_f_len=256,
        imu_dropout=0.1, seq_len=args.seq_len
    )
    model = FeatureEncodingModel(params)

    # Load pre-trained weights
    print(f"[INFO] Loading pre-trained weights from: {args.weights}")
    pretrained_w = torch.load(args.weights, map_location='cpu')
    model_dict = model.state_dict()
    update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
    
    print(f"[INFO] Matched {len(update_dict)}/{len(model_dict)} weight keys")
    if len(update_dict) != len(model_dict):
        missing = set(model_dict.keys()) - set(update_dict.keys())
        print(f"[WARNING] Missing keys: {missing}")
        print("[WARNING] Proceeding with partial weight loading...")
    
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)

    # Freeze encoder — no gradients needed, just inference
    for param in model.parameters():
        param.requires_grad = False

    # Move to GPU and set eval mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded on {device}, starting feature caching...")

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Cache features
    with torch.no_grad():
        for i, ((imgs, imus, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
            imgs = imgs.to(device).float()
            imus = imus.to(device).float()

            feat_v, feat_i = model(imgs, imus)
            latent_vector = torch.cat((feat_v, feat_i), 2)  # (1, seq_len, 768)
            latent_vector = latent_vector.squeeze(0)          # (seq_len, 768)

            # Save latent vector, ground truth, rotation, and weight
            np.save(os.path.join(save_dir, f"{i}.npy"), latent_vector.cpu().numpy())
            np.save(os.path.join(save_dir, f"{i}_gt.npy"), gts.cpu().numpy())
            np.save(os.path.join(save_dir, f"{i}_rot.npy"), rot.cpu().numpy())
            np.save(os.path.join(save_dir, f"{i}_w.npy"), w.cpu().numpy())

    print(f"[DONE] Cached {i+1} samples to {save_dir}")
    print(f"[INFO] Each sample shape: ({args.seq_len}, 768) = visual(512) + inertial(256)")


if __name__ == '__main__':
    main()
