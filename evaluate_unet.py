from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent

SPLIT_DIR = ROOT / "prepared" / "dataset_split"
OUTPUT_DIR = ROOT / "prepared" / "unet_baseline"

CHECKPOINT = OUTPUT_DIR / "best_unet.pth"
VAL_CSV = SPLIT_DIR / "validation_manifest.csv"

DEVICE = torch.device("cpu")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
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

        self.up4 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec4 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(
            32, 16, 2, stride=2
        )
        self.dec1 = DoubleConv(32, 16)

        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.dec4(
            torch.cat(
                [self.up4(b), e4],
                dim=1
            )
        )

        d3 = self.dec3(
            torch.cat(
                [self.up3(d4), e3],
                dim=1
            )
        )

        d2 = self.dec2(
            torch.cat(
                [self.up2(d3), e2],
                dim=1
            )
        )

        d1 = self.dec1(
            torch.cat(
                [self.up1(d2), e1],
                dim=1
            )
        )

        return self.output(d1)


def resolve_path(value):

    path = Path(str(value))

    if path.is_absolute():
        return path

    return ROOT / path


def calculate_metrics(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(
        pred,
        gt
    ).sum()

    fp = np.logical_and(
        pred,
        ~gt
    ).sum()

    fn = np.logical_and(
        ~pred,
        gt
    ).sum()

    tn = np.logical_and(
        ~pred,
        ~gt
    ).sum()

    epsilon = 1e-8

    dice = (
        2 * tp
        /
        (2 * tp + fp + fn + epsilon)
    )

    iou = (
        tp
        /
        (tp + fp + fn + epsilon)
    )

    precision = (
        tp
        /
        (tp + fp + epsilon)
    )

    recall = (
        tp
        /
        (tp + fn + epsilon)
    )

    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn + epsilon)
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():

    print("===== U-NET VALIDATION EVALUATION =====")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    if not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Validation manifest not found: {VAL_CSV}"
        )

    df = pd.read_csv(VAL_CSV)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model = UNet().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    size = checkpoint["image_size"]

    results = []

    for i, row in df.iterrows():

        image_path = resolve_path(
            row["image_path"]
        )

        mask_path = resolve_path(
            row["mask_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        image = image.resize(
            (size, size),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (size, size),
            Image.Resampling.NEAREST
        )

        image_array = (
            np.asarray(
                image,
                dtype=np.float32
            )
            / 255.0
        )

        gt = (
            np.asarray(mask)
            > 127
        )

        tensor = torch.from_numpy(
            image_array.copy()
        )

        tensor = tensor.permute(
            2, 0, 1
        )

        tensor = tensor.unsqueeze(0)

        with torch.no_grad():

            logits = model(
                tensor
            )

            probabilities = torch.sigmoid(
                logits
            )[0, 0]

            prediction = (
                probabilities >= 0.5
            ).cpu().numpy()

        metrics = calculate_metrics(
            prediction,
            gt
        )

        results.append({
            "filename":
                row["corrected_filename"],

            "dice":
                metrics["dice"],

            "iou":
                metrics["iou"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "accuracy":
                metrics["accuracy"],

            "tp":
                metrics["tp"],

            "fp":
                metrics["fp"],

            "fn":
                metrics["fn"],

            "tn":
                metrics["tn"],
        })

        print(
            f"{i + 1:02d}/"
            f"{len(df)} | "
            f"Dice={metrics['dice']:.4f} | "
            f"IoU={metrics['iou']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f}"
        )

    results_df = pd.DataFrame(results)

    report_path = (
        OUTPUT_DIR /
        "validation_metrics.csv"
    )

    results_df.to_csv(
        report_path,
        index=False
    )

    print()
    print("===== VALIDATION RESULTS =====")

    print(
        f"Mean Dice:      "
        f"{results_df['dice'].mean():.4f}"
    )

    print(
        f"Mean IoU:       "
        f"{results_df['iou'].mean():.4f}"
    )

    print(
        f"Mean Precision: "
        f"{results_df['precision'].mean():.4f}"
    )

    print(
        f"Mean Recall:    "
        f"{results_df['recall'].mean():.4f}"
    )

    print(
        f"Mean Accuracy:  "
        f"{results_df['accuracy'].mean():.4f}"
    )

    print()
    print(
        "Best Dice:       "
        f"{results_df['dice'].max():.4f}"
    )

    print(
        "Worst Dice:      "
        f"{results_df['dice'].min():.4f}"
    )

    print()
    print(
        "Saved report:"
    )

    print(report_path)


if __name__ == "__main__":
    main()