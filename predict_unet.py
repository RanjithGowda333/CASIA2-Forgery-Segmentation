
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "prepared" / "unet_baseline"
SPLIT_DIR = ROOT / "prepared" / "dataset_split"

CHECKPOINT = OUTPUT_DIR / "best_unet.pth"
VAL_CSV = SPLIT_DIR / "validation_manifest.csv"

DEVICE = torch.device("cpu")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.enc4 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)

def resolve_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    df = pd.read_csv(VAL_CSV)

    if not {"image_path", "mask_path"}.issubset(df.columns):
        raise ValueError(
            f"Expected image_path and mask_path columns; found {list(df.columns)}"
        )

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )

    model = UNet().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    size = checkpoint["image_size"]
    preview_dir = OUTPUT_DIR / "validation_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for index, row in df.head(6).iterrows():
        image_path = resolve_path(row["image_path"])
        mask_path = resolve_path(row["mask_path"])

        original = Image.open(image_path).convert("RGB")
        ground_truth = Image.open(mask_path).convert("L")

        resized = original.resize(
            (size, size), Image.Resampling.BILINEAR
        )
        image_array = np.asarray(resized, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(image_array.copy())
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            probabilities = torch.sigmoid(model(tensor))[0, 0]
            prediction = (probabilities >= 0.5).numpy().astype(np.uint8)

        gt = ground_truth.resize(
            (size, size), Image.Resampling.NEAREST
        )
        gt = np.asarray(gt) > 127

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(resized)
        axes[0].set_title("Original image")

        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground-truth mask")

        axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("U-Net prediction")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        output_path = preview_dir / f"sample_{index:03d}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        print("Saved:", output_path)

    print("\nPreview images saved to:", preview_dir)

if __name__ == "__main__":
    main()