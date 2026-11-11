
from pathlib import Path
import pandas as pd

BASE = Path.home() / "Downloads" / "CASIA2"
TP_DIR = BASE / "Tp"
GT_DIR = BASE / "CASIA 2 Groundtruth"
EXCEL = BASE / "fileNamesCorrection.xlsx"

df = pd.read_excel(
    EXCEL, sheet_name="Incorrect file names"
).dropna(subset=["Original file name", "Correct file name"])

def find_files(folder, filename):
    return [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.name.lower() == filename.lower()
    ]

for _, row in df.head(10).iterrows():
    original = str(row["Original file name"]).strip()
    corrected = str(row["Correct file name"]).strip()

    print("\nOriginal: ", original)
    print("Corrected:", corrected)
    print("Original image found:", len(find_files(TP_DIR, original)))
    print("Corrected image found:", len(find_files(TP_DIR, corrected)))

    expected_mask = Path(corrected).stem + "_gt.png"
    print("Expected mask:", expected_mask)
    print("Mask found:", len(find_files(GT_DIR, expected_mask)))
