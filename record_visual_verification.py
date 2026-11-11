
from pathlib import Path
from datetime import date
import pandas as pd

# Project paths
BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "prepared" / "verified_dataset" / "verified_manifest.csv"
OUTPUT = BASE / "prepared" / "verified_dataset" / "visual_verification_record.csv"

EXPECTED_PAIRS = 141

# Read the existing manifest
if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)

required = {
    "original_filename",
    "corrected_filename",
    "image_path",
    "mask_path",
    "review_decision",
}

missing_columns = required - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing manifest columns: {sorted(missing_columns)}")

if len(df) != EXPECTED_PAIRS:
    raise ValueError(
        f"Expected {EXPECTED_PAIRS} pairs, found {len(df)}. "
        "Check the manifest before proceeding."
    )

# Confirm that each pair is present and has a MATCH decision.
if df["original_filename"].isna().any():
    raise ValueError("Some original filenames are missing.")

if df["image_path"].isna().any() or df["mask_path"].isna().any():
    raise ValueError("Some image or mask paths are missing.")

if not df["review_decision"].astype(str).str.upper().eq("MATCH").all():
    raise ValueError("Some pairs do not have a MATCH decision.")

missing_images = [
    p for p in df["image_path"].astype(str)
    if not Path(p).is_file()
]
missing_masks = [
    p for p in df["mask_path"].astype(str)
    if not Path(p).is_file()
]

if missing_images or missing_masks:
    raise FileNotFoundError(
        f"Missing images: {len(missing_images)}; "
        f"missing masks: {len(missing_masks)}. "
        "Fix these paths before recording verification."
    )

# Record the user's completed visual inspection.
# This does not independently inspect or prove mask correctness.
df["visual_review_status"] = "VISUALLY_VERIFIED"
df["visual_review_method"] = "Human inspection of image-mask overlays"
df["visual_review_date"] = date.today().isoformat()
df["visual_review_notes"] = (
    "User confirmed the overlays match manipulated regions."
)

# Save a separate report; original files remain unchanged.
df.to_csv(OUTPUT, index=False)

print("VISUAL VERIFICATION RECORD CREATED")
print("Pairs recorded:", len(df))
print("Images present:", len(df) - len(missing_images))
print("Masks present:", len(df) - len(missing_masks))
print("Review status:", df["visual_review_status"].iloc[0])
print("Saved to:", OUTPUT)
print("Original images, masks, and manifest were not modified.")
