
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
report_path = (
    base / "prepared" / "correction_matching_report_v2.csv"
)

if not report_path.exists():
    raise FileNotFoundError(
        f"Report not found: {report_path}"
    )

df = pd.read_csv(report_path)

unresolved = df[df["status"] == "UNRESOLVED"]

print("Unresolved rows:", len(unresolved))
print()

for _, row in unresolved.iterrows():
    print("Original:", row["original_filename"])
    print("Corrected:", row["corrected_filename"])
    print("Image matches:", row["image_match_count"])
    print("Mask matches:", row["mask_match_count"])
    print("-" * 60)

output = base / "prepared" / "unresolved_rows.csv"
unresolved.to_csv(output, index=False)

print("\nSaved unresolved rows to:", output)
