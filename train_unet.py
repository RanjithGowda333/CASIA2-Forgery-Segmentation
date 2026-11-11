
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "prepared" / "dataset_split"
OUTPUT_DIR = ROOT / "prepared" / "unet_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLIT_DIR / "train_manifest.csv"
VAL_CSV = SPLIT_DIR / "validation_manifest.csv"

IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42

DEVICE = torch.device("cpu")


# ---------------- REPRODUCIBILITY ----------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)
print("Project:", ROOT)


# ---------------- MANIFEST READER ----------------

def read_manifest(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\n{csv_path.name} columns:", list(df.columns))

    # Recognize common manifest column names.
    image_candidates = [
        "image_path", "image", "image_file", "image_filename",
        "tampered_image", "image_filepath"
    ]
    mask_candidates = [
        "mask_path", "mask", "mask_file", "mask_filename",
        "groundtruth_path", "ground_truth", "mask_filepath"
    ]

    lower_columns = {str(c).lower(): c for c in df.columns}

    image_col = next(
        (lower_columns[c] for c in image_candidates
         if c in lower_columns), None
    )
    mask_col = next(
        (lower_columns[c] for c in mask_candidates
         if c in lower_columns), None
    )

    if image_col is None or mask_col is None:
        raise ValueError(
            "Could not identify image/mask columns. "
            f"CSV columns are: {list(df.columns)}. "
            "Update image_candidates and mask_candidates "
            "to match your manifest."
        )

    records = []

    for _, row in df.iterrows():
        image_path = Path(str(row[image_col]))
        mask_path = Path(str(row[mask_col]))

        # Relative paths are resolved from the project folder.
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        if not mask_path.is_absolute():
            mask_path = ROOT / mask_path

        if not image_path.is_file():
            raise FileNotFoundError(f"Image missing: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask missing: {mask_path}")

        records.append((image_path, mask_path))

    if not records:
        raise ValueError(f"No image-mask pairs in {csv_path}")

    return records


# ---------------- DATASET ----------------

class TamperingDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, mask_path = self.records[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Bilinear interpolation for images.
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # Nearest-neighbor interpolation preserves mask labels.
        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = np.asarray(mask, dtype=np.uint8)

        # CASIA masks are expected to be binary: 0 and 255.
        labels = np.unique(mask)
        if not np.isin(labels, [0, 255]).all():
            raise ValueError(
                f"Unexpected mask labels {labels} in {mask_path}"
            )

        # RGB image: HWC -> CHW.
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)

        # White mask pixels represent manipulated regions.
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask.copy()).unsqueeze(0)

        return image, mask


# ---------------- U-NET ----------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=1,
                bias=False
            ),
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

        # Raw logits; apply sigmoid only for metrics/inference.
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


# ---------------- LOSS AND METRICS ----------------

def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    probabilities = torch.sigmoid(logits)
    dims = (1, 2, 3)

    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1 - dice.mean()

    return bce + dice_loss


def calculate_metrics(logits, targets):
    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5

    predictions = predictions.bool()
    truth = truth.bool()

    tp = (predictions & truth).sum().item()
    fp = (predictions & ~truth).sum().item()
    fn = (~predictions & truth).sum().item()

    return tp, fp, fn


def metrics_from_counts(tp, fp, fn):
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


# ---------------- TRAIN / VALIDATE ----------------

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    tp_total = fp_total = fn_total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = combined_loss(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            tp, fp, fn = calculate_metrics(logits.detach(), masks)
            tp_total += tp
            fp_total += fp
            fn_total += fn

    metrics = metrics_from_counts(tp_total, fp_total, fn_total)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    train_records = read_manifest(TRAIN_CSV)
    val_records = read_manifest(VAL_CSV)

    print(f"\nTraining pairs: {len(train_records)}")
    print(f"Validation pairs: {len(val_records)}")

    train_loader = DataLoader(
        TamperingDataset(train_records),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        TamperingDataset(val_records),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    best_dice = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val Dice={val_metrics['dice']:.4f} | "
            f"val IoU={val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "val_dice": best_dice,
                },
                OUTPUT_DIR / "best_unet.pth"
            )

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    print("\nTraining finished.")
    print("Best validation Dice:", round(best_dice, 4))
    print("Saved model:", OUTPUT_DIR / "best_unet.pth")
    print("Saved history:", OUTPUT_DIR / "training_history.csv")


if __name__ == "__main__":
    main()
