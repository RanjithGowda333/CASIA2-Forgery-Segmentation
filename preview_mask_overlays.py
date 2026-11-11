
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "prepared" / "verified_dataset" / "verified_manifest.csv"
OUT = BASE / "prepared" / "mask_overlay_preview"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

# Select six examples spread across the manifest.
indices = sorted(set(
    np.linspace(0, len(df) - 1, min(6, len(df)), dtype=int)
))

for number, idx in enumerate(indices, start=1):
    row = df.iloc[idx]

    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if image.size != mask.size:
        raise ValueError(f"Image/mask size mismatch: {image_path.name}")

    image_array = np.array(image)
    mask_array = np.array(mask)

    # Mark white-mask pixels in red.
    overlay_array = image_array.copy()
    selected = mask_array == 255
    overlay_array[selected] = (
        0.5 * image_array[selected]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    overlay = Image.fromarray(overlay_array)

    # Display original, mask, and overlay side by side.
    panels = [
        ("Original", image),
        ("Mask", Image.merge(
            "RGB", (mask, mask, mask)
        )),
        ("Overlay", overlay),
    ]

    thumb_w, thumb_h = 320, 240
    header_h = 30
    sheet = Image.new(
        "RGB", (thumb_w * 3, thumb_h + header_h), "white"
    )
    draw = ImageDraw.Draw(sheet)

    for col, (label, panel) in enumerate(panels):
        panel.thumbnail((thumb_w, thumb_h))
        x = col * thumb_w
        draw.text((x + 8, 8), label, fill="black")
        sheet.paste(panel, (x, header_h))

    output_path = OUT / f"sample_{number:02d}.jpg"
    sheet.save(output_path, quality=95)
    print("Saved:", output_path)

print("\nOpen the preview folder and inspect all six samples.")
