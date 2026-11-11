
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
SPLIT_DIR = BASE / "prepared" / "dataset_split"

FILES = {
    "train": SPLIT_DIR / "train_manifest.csv",
    "validation": SPLIT_DIR / "validation_manifest.csv",
    "test": SPLIT_DIR / "test_manifest.csv",
}

def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def source_group(filename):
    # Extract source identifiers using the same
    # filename convention used by the split script.
    parts = Path(str(filename)).stem.split("_")
    if len(parts) < 8 or parts[0].lower() != "tp":
        return None
    return (parts[-3].lower(), parts[-2].lower())

def main():
    frames = {}

    for name, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        image_col = find_column(
            df, ["image_path", "image", "image_filename"]
        )
        mask_col = find_column(
            df, ["mask_path", "mask", "mask_filename"]
        )
        filename_col = find_column(
            df, ["original_filename", "corrected_filename",
                 "filename", "image_filename"]
        )

        if not image_col or not mask_col:
            raise ValueError(
                f"{name}: could not identify image/mask columns. "
                f"Columns found: {list(df.columns)}"
            )

        if not filename_col:
            raise ValueError(
                f"{name}: could not identify filename column. "
                f"Columns found: {list(df.columns)}"
            )

        df["_image_path"] = df[image_col].astype(str)
        df["_mask_path"] = df[mask_col].astype(str)
        df["_filename"] = df[filename_col].astype(str)
        frames[name] = df

    print("\n===== DATASET SPLIT VALIDATION =====")

    all_images = set()
    all_masks = set()
    all_groups = {}

    expected = {"train": 98, "validation": 21, "test": 22}

    for name, df in frames.items():
        missing_images = [
            p for p in df["_image_path"]
            if not Path(p).is_file()
        ]
        missing_masks = [
            p for p in df["_mask_path"]
            if not Path(p).is_file()
        ]

        images = set(df["_image_path"].str.lower())
        masks = set(df["_mask_path"].str.lower())

        print(f"\n{name.upper()}")
        print("Rows:", len(df))
        print("Expected rows:", expected[name])
        print("Missing images:", len(missing_images))
        print("Missing masks:", len(missing_masks))
        print("Duplicate image paths:", len(df) - len(images))
        print("Duplicate mask paths:", len(df) - len(masks))

        all_images.update(images)
        all_masks.update(masks)

        groups = set()
        for filename in df["_filename"]:
            group = source_group(filename)
            if group is not None:
                groups.add(group)

        all_groups[name] = groups
        print("Recognized source groups:", len(groups))

    print("\n===== CROSS-SPLIT CHECKS =====")

    names = ["train", "validation", "test"]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            df_a, df_b = frames[a], frames[b]

            image_overlap = (
                set(df_a["_image_path"].str.lower())
                & set(df_b["_image_path"].str.lower())
            )
            mask_overlap = (
                set(df_a["_mask_path"].str.lower())
                & set(df_b["_mask_path"].str.lower())
            )
            group_overlap = all_groups[a] & all_groups[b]

            print(f"\n{a} vs {b}")
            print("Shared image paths:", len(image_overlap))
            print("Shared mask paths:", len(mask_overlap))
            print("Shared source groups:", len(group_overlap))

    total = sum(len(df) for df in frames.values())
    print("\nTotal split rows:", total)
    print("Unique image paths:", len(all_images))
    print("Unique mask paths:", len(all_masks))

    if total != 141:
        print("WARNING: Expected 141 total rows.")
    else:
        print("Pair count: PASS")

    print("\nValidation finished. Review all warnings above.")

if __name__ == "__main__":
    main()
