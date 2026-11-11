
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

BASE = Path.home() / "Downloads" / "CASIA2"
REPORT = BASE / "prepared" / "size_mismatches.csv"

df = pd.read_csv(REPORT)

print("Mismatched pairs:", len(df))
print(df[["image", "mask"]].head())

# Display up to 4 mismatched pairs for manual inspection
for _, row in df.head(4).iterrows():
    image_path = Path(row["image"])
    mask_path = Path(row["mask"])

    with Image.open(image_path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")

    with Image.open(mask_path) as im:
        mask = im.convert("L")

    image_np = np.array(image)
    mask_np = np.array(mask)

    print("\nImage:", image_path.name)
    print("Image size:", image.size)
    print("Mask:", mask_path.name)
    print("Mask size:", mask.size)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Tampered image")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Ground-truth mask")

    # Show overlay only if dimensions match
    if image.size == mask.size:
        axes[2].imshow(image_np)
        axes[2].imshow(mask_np > 0, cmap="Reds", alpha=0.4)
        axes[2].set_title("Overlay")
    else:
        axes[2].text(
            0.5, 0.5,
            "Dimensions differ\nOverlay not displayed",
            ha="center", va="center"
        )
        axes[2].set_title("Needs investigation")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
