
from pathlib import Path
import pandas as pd

BASE = Path.home() / "Downloads" / "CASIA2"
file_path = BASE / "fileNamesCorrection.xlsx"

if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

excel = pd.ExcelFile(file_path)

print("Sheets:", excel.sheet_names)

for sheet in excel.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)

    print(f"\n--- Sheet: {sheet} ---")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(10).to_string(index=False))
