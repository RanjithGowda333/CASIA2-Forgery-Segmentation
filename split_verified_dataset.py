
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(__file__).resolve().parent
DATASET = BASE / "prepared" / "verified_dataset"
MANIFEST = DATASET / "verified_manifest.csv"
OUT = BASE / "prepared" / "dataset_split"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

# Use the verified image and mask filenames.
required = [
    "original_filename",
    "image_path",
    "mask_path",
    "review_decision",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing manifest columns: {missing}")

if len(df) != 141:
    raise ValueError(f"Expected 141 records, found {len(df)}")

# Validate each image-mask pair before splitting.
for _, row in df.iterrows():
    if not Path(row["image_path"]).is_file():
        raise FileNotFoundError(row["image_path"])
    if not Path(row["mask_path"]).is_file():
        raise FileNotFoundError(row["mask_path"])

if not df["review_decision"].astype(str).str.upper().eq("MATCH").all():
    raise ValueError("Some review decisions are not MATCH.")

# Preserve the existing source-ID extraction logic.
def get_source_ids(filename):
    parts = Path(str(filename)).stem.split("_")

    if len(parts) < 8 or parts[0].lower() != "tp":
        return None

    return parts[-3].lower(), parts[-2].lower()

source_ids = df["original_filename"].apply(get_source_ids)

if source_ids.isna().any():
    bad = df.loc[source_ids.isna(), "original_filename"].tolist()
    raise ValueError(
        f"Could not extract source IDs from {len(bad)} filenames. "
        f"Examples: {bad[:5]}"
    )

# Combine both IDs into one group label.
df["source_group"] = source_ids.apply(
    lambda pair: f"{pair[0]}_{pair[1]}"
)

# Split groups: approximately 70% train, 30% remainder.
first_split = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)
train_idx, remaining_idx = next(
    first_split.split(df, groups=df["source_group"])
)

train = df.iloc[train_idx].copy()
remaining = df.iloc[remaining_idx].copy()

# Split remainder approximately equally into validation and test.
second_split = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=42
)
val_idx, test_idx = next(
    second_split.split(
        remaining, groups=remaining["source_group"]
    )
)

validation = remaining.iloc[val_idx].copy()
test = remaining.iloc[test_idx].copy()

# Confirm there is no source-group leakage.
train_groups = set(train["source_group"])
val_groups = set(validation["source_group"])
test_groups = set(test["source_group"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# Save split manifests. Original files remain untouched.
for name, part in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    part.to_csv(OUT / f"{name}_manifest.csv", index=False)

print("\nDATASET SPLIT COMPLETE")
print("Total pairs:", len(df))

for name, part in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:
    print(
        f"{name}: {len(part)} images, "
        f"{part['source_group'].nunique()} source groups "
        f"({len(part) / len(df) * 100:.1f}%)"
    )

print("\nSource-group overlaps:")
print("Train/validation:", len(train_groups & val_groups))
print("Train/test:", len(train_groups & test_groups))
print("Validation/test:", len(val_groups & test_groups))

print("\nSaved split manifests to:", OUT)
print("Original images and masks were not modified.")
