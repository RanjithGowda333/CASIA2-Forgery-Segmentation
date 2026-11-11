
from pathlib import Path
from collections import defaultdict
from PIL import Image
import pandas as pd
import numpy as np
import random

# --------------------------------------------------
# 1. SET YOUR DATASET PATH
# --------------------------------------------------
BASE = Path.home() / "Downloads" / "CASIA2"

AU_DIR = BASE / "Au"
TP_DIR = BASE / "Tp"
GT_DIR = BASE / "CASIA 2 Groundtruth"

OUTPUT_DIR = BASE / "prepared"
AUTH_MASK_DIR = OUTPUT_DIR / "authentic_masks"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUTH_MASK_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

# --------------------------------------------------
# 2. CHECK THE FOLDERS
# --------------------------------------------------
for folder in [AU_DIR, TP_DIR, GT_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder not found: {folder}\n"
            "Check the path and folder name."
        )

print("Dataset folders found successfully.")

# --------------------------------------------------
# 3. FIND IMAGE AND MASK FILES
# --------------------------------------------------
def get_files(folder, extensions):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

au_files = get_files(AU_DIR, IMAGE_EXTS)
tp_files = get_files(TP_DIR, IMAGE_EXTS)
gt_files = get_files(GT_DIR, {".png"})

print("Authentic image files:", len(au_files))
print("Tampered image files:", len(tp_files))
print("Ground-truth PNG files:", len(gt_files))

# --------------------------------------------------
# 4. BUILD THE MASK INDEX
#    Supports name.png and name_gt.png
# --------------------------------------------------
def normalize_mask_stem(path):
    stem = path.stem.lower()

    if stem.endswith("_gt"):
        stem = stem[:-3]

    return stem

mask_index = defaultdict(list)

for mask_path in gt_files:
    mask_index[normalize_mask_stem(mask_path)].append(mask_path)

# --------------------------------------------------
# 5. MATCH TAMPERED IMAGES TO MASKS
# --------------------------------------------------
matched_rows = []
missing_masks = []
ambiguous_masks = []
invalid_files = []
size_mismatches = []
empty_masks = []

for image_path in tp_files:
    candidates = mask_index.get(image_path.stem.lower(), [])

    if len(candidates) == 0:
        missing_masks.append(str(image_path))
        continue

    if len(candidates) > 1:
        ambiguous_masks.append({
            "image": str(image_path),
            "masks": " | ".join(str(p) for p in candidates)
        })
        continue

    mask_path = candidates[0]

    try:
        with Image.open(image_path) as im:
            im.verify()

        with Image.open(mask_path) as im:
            im.verify()

        with Image.open(image_path) as im:
            image_size = im.size

        with Image.open(mask_path) as im:
            mask_size = im.size
            mask = np.asarray(im.convert("L"))

        if image_size != mask_size:
            size_mismatches.append({
                "image": str(image_path),
                "mask": str(mask_path),
                "image_size": str(image_size),
                "mask_size": str(mask_size)
            })
            continue

        if mask.max() == 0:
            empty_masks.append({
                "image": str(image_path),
                "mask": str(mask_path)
            })
            continue

        matched_rows.append({
            "image_path": str(image_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "label": 1,
            "image_name": image_path.name,
            "mask_name": mask_path.name,
            "split": ""
        })

    except Exception as error:
        invalid_files.append({
            "image": str(image_path),
            "mask": str(mask_path),
            "error": str(error)
        })

# --------------------------------------------------
# 6. CREATE BLACK MASKS FOR AUTHENTIC IMAGES
# --------------------------------------------------
authentic_rows = []

for image_path in au_files:
    try:
        with Image.open(image_path) as im:
            im.verify()

        with Image.open(image_path) as im:
            width, height = im.size

        # Use relative path to avoid filename collisions
        relative = image_path.relative_to(AU_DIR)
        mask_relative = relative.with_suffix(".png")
        mask_path = AUTH_MASK_DIR / mask_relative

        mask_path.parent.mkdir(parents=True, exist_ok=True)

        # Save a black mask without changing the original image
        Image.fromarray(
            np.zeros((height, width), dtype=np.uint8)
        ).save(mask_path)

        authentic_rows.append({
            "image_path": str(image_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "label": 0,
            "image_name": image_path.name,
            "mask_name": mask_path.name,
            "split": ""
        })

    except Exception as error:
        invalid_files.append({
            "image": str(image_path),
            "mask": "",
            "error": str(error)
        })

# --------------------------------------------------
# 7. CREATE MANIFEST
# --------------------------------------------------
rows = authentic_rows + matched_rows
manifest = pd.DataFrame(rows)

if manifest.empty:
    raise RuntimeError(
        "No usable samples found. Check the folder paths and filenames."
    )

manifest_path = OUTPUT_DIR / "manifest.csv"
manifest.to_csv(manifest_path, index=False)

# --------------------------------------------------
# 8. SAVE FILES THAT NEED ATTENTION
# --------------------------------------------------
def save_report(name, rows):
    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / name, index=False
        )

save_report("missing_masks.csv", [
    {"image": p} for p in missing_masks
])

save_report("ambiguous_masks.csv", ambiguous_masks)
save_report("size_mismatches.csv", size_mismatches)
save_report("empty_masks.csv", empty_masks)
save_report("invalid_files.csv", invalid_files)

# --------------------------------------------------
# 9. PRINT SUMMARY
# --------------------------------------------------
print("\n========== DATASET SUMMARY ==========")
print("Authentic samples:", len(authentic_rows))
print("Matched tampered samples:", len(matched_rows))
print("Total manifest rows:", len(manifest))
print("Missing masks:", len(missing_masks))
print("Ambiguous matches:", len(ambiguous_masks))
print("Image-mask size mismatches:", len(size_mismatches))
print("All-black tampered masks:", len(empty_masks))
print("Unreadable files:", len(invalid_files))
print("\nManifest saved to:")
print(manifest_path)

print("\nReview reports saved in:")
print(OUTPUT_DIR)
