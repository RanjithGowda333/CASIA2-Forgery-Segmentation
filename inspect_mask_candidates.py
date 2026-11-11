
from pathlib import Path
import pandas as pd
import re

base = Path(__file__).resolve().parent

unresolved_path = base / "prepared" / "unresolved_rows.csv"
mask_dir = base / "CASIA 2 Groundtruth"

if not unresolved_path.exists():
    raise FileNotFoundError(f"Missing report: {unresolved_path}")

if not mask_dir.exists():
    raise FileNotFoundError(f"Missing mask folder: {mask_dir}")

df = pd.read_csv(unresolved_path)

mask_files = [
    p for p in mask_dir.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
]

# Index filenames by their final numeric ID.
mask_index = {}

for p in mask_files:
    match = re.search(r"_(\d+)_gt$", p.stem, re.IGNORECASE)
    if match:
        image_id = match.group(1)
        mask_index.setdefault(image_id, []).append(p)

print("Total ground-truth files:", len(mask_files))
print("Indexed GT masks:", sum(len(v) for v in mask_index.values()))

for _, row in df.iterrows():
    original = str(row["original_filename"]).strip()
    corrected = str(row["corrected_filename"]).strip()

    # Extract the final numeric ID before the extension.
    match = re.search(r"_(\d+)$", Path(corrected).stem)

    print("\n" + "=" * 70)
    print("Original: ", original)
    print("Corrected:", corrected)

    if not match:
        print("STATUS: Could not extract final numeric ID")
        continue

    image_id = match.group(1)
    candidates = mask_index.get(image_id, [])

    # Require the prefix before the xxx identifier to match.
    corrected_stem = Path(corrected).stem
    prefix = re.sub(r"_xxx\d+_\d+$", "", corrected_stem)

    candidates = [
        p for p in candidates
        if re.sub(r"_xxx\d+_\d+_gt$", "", p.stem) == prefix
    ]

    print("Final ID:", image_id)
    print("Matching candidates:", len(candidates))

    for p in candidates:
        print(" ", p.name)
        print("  Full path:", p.resolve())

    if len(candidates) == 1:
        print("STATUS: ONE CANDIDATE - needs visual verification")
    elif len(candidates) > 1:
        print("STATUS: AMBIGUOUS - multiple candidates")
    else:
        print("STATUS: NO MATCH - inspect naming rules")

print("\nInvestigation complete. No files were changed.")
