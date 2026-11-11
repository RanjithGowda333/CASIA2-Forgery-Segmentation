
from pathlib import Path
import ast
import pandas as pd
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
DATASET = BASE / "prepared" / "verified_dataset"
MANIFEST = DATASET / "verified_manifest.csv"
REPORT = DATASET / "integrity_check_report.csv"

def parse_size(value):
    """Convert a stored size such as '(240, 180)' into a tuple."""
    if isinstance(value, str):
        return tuple(ast.literal_eval(value))
    return tuple(value)

def check_dataset():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    required = [
        "original_filename", "corrected_filename",
        "image_path", "mask_path", "image_size",
        "mask_size", "mask_pixel_values", "review_decision"
    ]

    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    results = []

    for index, row in df.iterrows():
        image_path = Path(str(row["image_path"]))
        mask_path = Path(str(row["mask_path"]))

        image_exists = image_path.is_file()
        mask_exists = mask_path.is_file()
        dimensions_match = False
        recorded_sizes_match = False
        mask_values_valid = False
        error = ""

        if image_exists and mask_exists:
            try:
                with Image.open(image_path) as img:
                    actual_image_size = img.size

                with Image.open(mask_path) as mask:
                    actual_mask_size = mask.size
                    mask_array = np.array(mask.convert("L"))

                dimensions_match = (
                    actual_image_size == actual_mask_size
                )

                recorded_sizes_match = (
                    actual_image_size == parse_size(row["image_size"])
                    and actual_mask_size == parse_size(row["mask_size"])
                )

                unique_values = set(
                    np.unique(mask_array).tolist()
                )
                mask_values_valid = unique_values.issubset({0, 255})

            except Exception as exc:
                error = str(exc)

        decision_match = (
            str(row["review_decision"]).strip().upper() == "MATCH"
        )

        passed = (
            image_exists
            and mask_exists
            and dimensions_match
            and recorded_sizes_match
            and mask_values_valid
            and decision_match
            and not error
        )

        results.append({
            "row": index + 1,
            "original_filename": row["original_filename"],
            "image_exists": image_exists,
            "mask_exists": mask_exists,
            "dimensions_match": dimensions_match,
            "recorded_sizes_match": recorded_sizes_match,
            "mask_values_valid": mask_values_valid,
            "review_decision_match": decision_match,
            "status": "PASS" if passed else "FAIL",
            "error": error
        })

    report = pd.DataFrame(results)

    image_paths = df["image_path"].astype(str)
    mask_paths = df["mask_path"].astype(str)

    duplicate_images = int(image_paths.duplicated().sum())
    duplicate_masks = int(mask_paths.duplicated().sum())

    report.to_csv(REPORT, index=False)

    print("\n===== CASIA2 DATASET INTEGRITY CHECK =====")
    print("Manifest rows:", len(df))
    print("Pairs passed:", int((report["status"] == "PASS").sum()))
    print("Pairs failed:", int((report["status"] == "FAIL").sum()))
    print("Missing images:", int((~report["image_exists"]).sum()))
    print("Missing masks:", int((~report["mask_exists"]).sum()))
    print("Dimension mismatches:",
          int((~report["dimensions_match"]).sum()))
    print("Recorded size mismatches:",
          int((~report["recorded_sizes_match"]).sum()))
    print("Invalid mask pixel values:",
          int((~report["mask_values_valid"]).sum()))
    print("Non-MATCH review decisions:",
          int((~report["review_decision_match"]).sum()))
    print("Duplicate image paths:", duplicate_images)
    print("Duplicate mask paths:", duplicate_masks)
    print("Detailed report:", REPORT)

    all_pass = (
        len(df) == 141
        and (report["status"] == "PASS").all()
        and duplicate_images == 0
        and duplicate_masks == 0
    )

    print("\nFINAL RESULT:",
          "ALL CHECKS PASSED" if all_pass else "REVIEW REQUIRED")

if __name__ == "__main__":
    check_dataset()
