
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
TP_DIR = BASE / "Tp"
GT_DIR = BASE / "CASIA 2 Groundtruth"
EXCEL = BASE / "fileNamesCorrection.xlsx"
OUT = BASE / "prepared"

OUT.mkdir(exist_ok=True)

corrections = pd.read_excel(
    EXCEL, sheet_name="Incorrect file names"
).dropna(subset=["Original file name", "Correct file name"])

# Index actual files by their exact filenames.
image_index = {}
for p in TP_DIR.rglob("*"):
    if p.is_file():
        image_index.setdefault(p.name.lower(), []).append(p)

mask_index = {}
for p in GT_DIR.rglob("*"):
    if p.is_file() and p.suffix.lower() == ".png":
        mask_index.setdefault(p.name.lower(), []).append(p)

rows = []

for _, r in corrections.iterrows():
    original = Path(str(r["Original file name"]).strip()).name
    corrected = Path(str(r["Correct file name"]).strip()).name

    # First locate the image using its original name.
    image_matches = image_index.get(original.lower(), [])
    image_source = "original"

    # If not found, try the corrected name.
    if not image_matches:
        image_matches = image_index.get(corrected.lower(), [])
        image_source = "corrected"

    # Match the ground-truth mask using the stable filename prefix
    # and final image ID, ignoring the differing xxx sequence number.
    corrected_stem = Path(corrected).stem
    parts = corrected_stem.split("_")

    if len(parts) >= 2 and parts[-2].startswith("xxx"):
        mask_stem = "_".join(parts[:-2] + ["xxx00000", parts[-1]])
        mask_name = mask_stem + "_gt.png"
    else:
        # Fallback to the original exact-name matching rule.
        mask_name = corrected_stem + "_gt.png"

    mask_matches = mask_index.get(mask_name.lower(), [])
    

    rows.append({
        "original_filename": original,
        "corrected_filename": corrected,
        "image_path": str(image_matches[0]) if len(image_matches) == 1 else "",
        "mask_path": str(mask_matches[0]) if len(mask_matches) == 1 else "",
        "image_match_count": len(image_matches),
        "mask_match_count": len(mask_matches),
        "image_match_source": image_source if image_matches else "not found",
        "status": (
            "CANDIDATE_REVIEW"
            if len(image_matches) == 1 and len(mask_matches) == 1
            else "UNRESOLVED"
        ),
    })

report = pd.DataFrame(rows)
report.to_csv(
    OUT / "correction_matching_report_v2.csv", index=False
)

print("Correction rows:", len(report))
print("Unique image matches:", (report["image_match_count"] == 1).sum())
print("Unique mask matches:", (report["mask_match_count"] == 1).sum())
print("Candidate pairs requiring review:", (report["status"] == "CANDIDATE_REVIEW").sum())
print("Unresolved rows:", (report["status"] == "UNRESOLVED").sum())
print("Saved:", OUT / "correction_matching_report_v2.csv")
