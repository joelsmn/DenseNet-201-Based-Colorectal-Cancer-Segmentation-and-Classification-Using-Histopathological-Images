"""
segmentation/unet.py
─────────────────────────────────────────────────────────────────────────────
Stage 3 — SEGMENTATION  (U-Net CNN)

Architecture
  • Encoder : 4 × (Conv-BN-ReLU) × 2  +  MaxPool
  • Bottleneck: Conv-BN-ReLU × 2
  • Decoder : 4 × UpConv + skip concat + (Conv-BN-ReLU) × 2
  • Output  : 1×1 Conv → sigmoid  (binary mask)

Training
  • Loss : Dice + BCE combined
  • Optimiser : Adam  lr=1e-3
  • 30 epochs, ReduceLROnPlateau
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import UNET_FILTERS, MODELS_DIR


# ─── Building blocks ──────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─── U-Net ────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net with 4 encoder / decoder levels.
    in_channels = 3 (RGB),  out_channels = 1 (binary mask).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 filters=None):
        super().__init__()
        if filters is None:
            filters = UNET_FILTERS   # [64, 128, 256, 512, 1024]

        # Encoder
        self.enc1 = DownBlock(in_channels, filters[0])
        self.enc2 = DownBlock(filters[0],  filters[1])
        self.enc3 = DownBlock(filters[1],  filters[2])
        self.enc4 = DownBlock(filters[2],  filters[3])

        # Bottleneck
        self.bottleneck = DoubleConv(filters[3], filters[4])

        # Decoder
        self.dec4 = UpBlock(filters[4], filters[3])
        self.dec3 = UpBlock(filters[3], filters[2])
        self.dec2 = UpBlock(filters[2], filters[1])
        self.dec1 = UpBlock(filters[1], filters[0])

        # Output
        self.out_conv = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return torch.sigmoid(self.out_conv(x))


# ─── Combined Dice + BCE loss ─────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCELoss()

    def forward(self, pred, target):
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / \
                        (pred_flat.sum() + target_flat.sum() + self.smooth)
        bce_loss  = self.bce(pred_flat, target_flat)
        return dice_loss + bce_loss


# ─── Training helper ──────────────────────────────────────────────────────

def train_unet(verbose: bool = True) -> UNet:
    """
    Full U-Net training loop.
    Returns the trained model.
    """
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import os

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config.config import (UNET_EPOCHS, UNET_LR, UNET_BATCH,
                                SEG_PRED_DIR, SEED)
    from utils.datasets import CancerSegmentationDataset

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    train_ds = CancerSegmentationDataset("train")
    val_ds   = CancerSegmentationDataset("val")
    train_dl = DataLoader(train_ds, batch_size=UNET_BATCH, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=UNET_BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)

    model     = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=UNET_LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=0.5, patience=4)

    best_val_loss = float("inf")
    best_path     = MODELS_DIR / "unet_best.pth"

    print(f"\n  Training U-Net for {UNET_EPOCHS} epochs")
    print(f"  Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")

    for epoch in range(1, UNET_EPOCHS + 1):
        # ── Train ───────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_dl, desc=f"  Epoch {epoch:3d}/{UNET_EPOCHS} [Train]",
                                leave=False, disable=not verbose):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()

        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)
        scheduler.step(val_loss)

        pct = epoch / UNET_EPOCHS * 100
        if verbose:
            print(f"  [{pct:5.1f}%]  Epoch {epoch:3d}  "
                  f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(best_path))

    print(f"\n✔  U-Net training complete.  Best val loss: {best_val_loss:.4f}")
    print(f"   Model saved → {best_path}")
    model.load_state_dict(torch.load(str(best_path), map_location=device))
    return model


# ─── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  STAGE 3 — SEGMENTATION (U-Net)")
    print("=" * 60)
    trained_model = train_unet(verbose=True)