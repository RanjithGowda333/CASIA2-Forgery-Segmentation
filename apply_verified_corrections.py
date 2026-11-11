
from pathlib import Path
import shutil
import pandas as pd
from PIL import Image

BASE = Path(__file__).resolve().parent
PREPARED = BASE / "prepared"

REVIEW_CSV = PREPARED / "visual_review_results.csv"
REPORT_CSV = PREPARED / "correction_matching_report_v2.csv"
OUTPUT = PREPARED / "verified_dataset"

EXPECTED_ROWS = 141


def fail(message):
    raise SystemExit(f"\nERROR: {message}")


def normalized_name(value):
    return Path(str(value).strip()).name.lower()


# 1. Check required reports.
for path in (REVIEW_CSV, REPORT_CSV):
    if not path.is_file():
        fail(f"Required file not found: {path}")

review = pd.read_csv(REVIEW_CSV)
report = pd.read_csv(REPORT_CSV)

required_review_columns = {
    "original_filename",
    "corrected_filename",
    "image_path",
    "mask_path",
    "dimensions_match",
    "review_decision",
}

missing_columns = required_review_columns - set(review.columns)
if missing_columns:
    fail(f"Review CSV is missing columns: {sorted(missing_columns)}")

required_report_columns = {
    "original_filename",
    "corrected_filename",
    "image_path",
    "mask_path",
    "status",
}

missing_columns = required_report_columns - set(report.columns)
if missing_columns:
    fail(f"Matching report is missing columns: {sorted(missing_columns)}")

# 2. Require all 141 pairs in both reports.
if len(review) != EXPECTED_ROWS or len(report) != EXPECTED_ROWS:
    fail(
        f"Expected {EXPECTED_ROWS} rows in both reports. "
        f"Review rows={len(review)}, matching rows={len(report)}"
    )

# 3. Match review decisions to the generated matching report.
review = review.copy()
report = report.copy()

for df in (review, report):
    df["original_key"] = df["original_filename"].map(normalized_name)
    df["corrected_key"] = df["corrected_filename"].map(normalized_name)

keys = ["original_key", "corrected_key"]

if review.duplicated(keys).any() or report.duplicated(keys).any():
    fail("Duplicate original/corrected filename pairs were found.")

if set(map(tuple, review[keys].values)) != set(
    map(tuple, report[keys].values)
):
    fail("The review CSV and matching report contain different pairs.")

# Ensure every pair was explicitly approved.
decisions = review["review_decision"].astype(str).str.strip().str.upper()

if not decisions.eq("MATCH").all():
    fail(
        "Some pairs are not marked MATCH. "
        "Review those rows before proceeding."
    )

if not review["dimensions_match"].astype(str).str.strip().str.upper().isin(
    ["TRUE", "1"]
).all():
    fail("At least one pair has a dimension mismatch.")

if not report["status"].astype(str).str.strip().eq(
    "CANDIDATE_REVIEW"
).all():
    fail("The matching report contains unresolved or unexpected rows.")

# 4. Validate every source image and mask before copying anything.
pairs = []
target_names = set()

for _, row in review.iterrows():
    original = str(row["original_filename"]).strip()
    corrected = Path(str(row["corrected_filename"]).strip()).name

    if not corrected or corrected in {".", ".."}:
        fail(f"Invalid corrected filename for {original}")

    target_key = corrected.lower()
    if target_key in target_names:
        fail(f"Duplicate corrected output filename: {corrected}")
    target_names.add(target_key)

    image_path = Path(str(row["image_path"]).strip())
    mask_path = Path(str(row["mask_path"]).strip())

    if not image_path.is_file():
        fail(f"Image is missing: {image_path}")

    if not mask_path.is_file():
        fail(f"Mask is missing: {mask_path}")

    try:
        with Image.open(image_path) as image:
            image_size = image.size

        with Image.open(mask_path) as mask:
            mask_size = mask.size
            mask_mode = mask.mode
            mask_format = mask.format
            mask_values = set(mask.convert("L").getdata())

    except Exception as exc:
        fail(f"Cannot read image/mask for {original}: {exc}")

    if image_size != mask_size:
        fail(
            f"Dimension mismatch for {original}: "
            f"image={image_size}, mask={mask_size}"
        )

    # CASIA2 binary ground-truth masks should normally use
    # black (0) and white (255). Stop if other values are present.
    if not mask_values.issubset({0, 255}):
        fail(
            f"Unexpected mask pixel values for {mask_path.name}. "
            "Inspect the mask before copying."
        )

    # Verify this source pair agrees with the matching report.
    match = report[
        (report["original_key"] == normalized_name(original))
        & (report["corrected_key"] == normalized_name(corrected))
    ]

    if len(match) != 1:
        fail(f"Cannot uniquely verify report entry for {original}")

    report_row = match.iloc[0]

    if (
        Path(str(report_row["image_path"]).strip()).resolve()
        != image_path.resolve()
        or Path(str(report_row["mask_path"]).strip()).resolve()
        != mask_path.resolve()
    ):
        fail(f"Source paths differ between reports for {original}")

    pairs.append({
        "original_filename": original,
        "corrected_filename": corrected,
        "image_path": image_path,
        "mask_path": mask_path,
        "image_size": image_size,
        "mask_size": mask_size,
        "mask_mode": mask_mode,
        "mask_format": mask_format,
        "mask_pixel_values": ",".join(map(str, sorted(mask_values))),
    })

# 5. Do not overwrite a previous output.
if OUTPUT.exists():
    fail(
        f"Output folder already exists: {OUTPUT}\n"
        "Rename or move that folder before creating a new dataset."
    )

# 6. Copy verified pairs to a separate dataset.
images_out = OUTPUT / "images"
masks_out = OUTPUT / "masks"
images_out.mkdir(parents=True)
masks_out.mkdir()

manifest_rows = []

try:
    for pair in pairs:
        image_dest = images_out / pair["corrected_filename"]
        mask_dest = masks_out / (
            Path(pair["corrected_filename"]).stem + "_gt.png"
        )

        if image_dest.exists() or mask_dest.exists():
            fail(f"Output filename collision for {pair['corrected_filename']}")

        shutil.copy2(pair["image_path"], image_dest)

        # Preserve the original mask format when copying the mask.
        # The destination filename provides an exact image-mask pairing.
        mask_extension = pair["mask_path"].suffix.lower()
        mask_dest = masks_out / (
            Path(pair["corrected_filename"]).stem + "_gt" + mask_extension
        )
        shutil.copy2(pair["mask_path"], mask_dest)

        manifest_rows.append({
            "original_filename": pair["original_filename"],
            "corrected_filename": pair["corrected_filename"],
            "image_path": str(image_dest),
            "mask_path": str(mask_dest),
            "image_size": str(pair["image_size"]),
            "mask_size": str(pair["mask_size"]),
            "mask_pixel_values": pair["mask_pixel_values"],
            "review_decision": "MATCH",
        })

    # 7. Re-open every copied pair and validate it.
    for row in manifest_rows:
        with Image.open(row["image_path"]) as image:
            image_size = image.size

        with Image.open(row["mask_path"]) as mask:
            mask_size = mask.size

        if image_size != mask_size:
            fail(f"Output dimension mismatch: {row['corrected_filename']}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT / "verified_manifest.csv", index=False)

except Exception:
    # Remove only the newly created output folder after a failed run.
    shutil.rmtree(OUTPUT, ignore_errors=True)
    raise

print("\nVERIFICATION AND COPY COMPLETE")
print("Verified pairs:", len(manifest_rows))
print("Images copied:", len(list(images_out.iterdir())))
print("Masks copied:", len(list(masks_out.iterdir())))
print("Missing source files: 0")
print("Dimension mismatches: 0")
print("Unreviewed pairs: 0")
print("Original CASIA2 files were not modified.")
print("Output dataset:", OUTPUT)
print("Manifest:", OUTPUT / "verified_manifest.csv")
