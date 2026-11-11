
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image

BASE = Path(__file__).resolve().parent
MASK_DIR = BASE / "prepared" / "verified_dataset" / "masks"
OUT = BASE / "prepared" / "mask_label_inspection.csv"

rows = []
global_counts = Counter()

for path in sorted(MASK_DIR.glob("*.png")):
    mask = np.array(Image.open(path).convert("L"))
    values, counts = np.unique(mask, return_counts=True)
    pixel_counts = dict(zip(values.tolist(), counts.tolist()))
    global_counts.update(pixel_counts)

    black = pixel_counts.get(0, 0)
    white = pixel_counts.get(255, 0)
    other = sum(n for v, n in pixel_counts.items()
                if v not in (0, 255))

    rows.append({
        "mask_filename": path.name,
        "width": mask.shape[1],
        "height": mask.shape[0],
        "black_pixels": black,
        "white_pixels": white,
        "other_pixels": other,
        "white_pixel_ratio": white / mask.size,
        "status": (
            "ALL_BLACK" if white == 0 else
            "ALL_WHITE" if black == 0 else
            "CHECK_OTHER_VALUES" if other else
            "BINARY_MASK"
        )
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("Masks inspected:", len(df))
print("Total pixel counts:", dict(sorted(global_counts.items())))
print("\nMask status summary:")
print(df["status"].value_counts().to_string())
print("\nWhite pixel ratio summary:")
print(df["white_pixel_ratio"].describe().to_string())
print("\nSaved report:", OUT)
