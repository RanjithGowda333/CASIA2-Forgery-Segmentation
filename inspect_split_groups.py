
from pathlib import Path
from collections import Counter
import pandas as pd

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "prepared" / "verified_dataset" / "verified_manifest.csv"

df = pd.read_csv(MANIFEST)

# Extract the two source-image identifiers from CASIA2-style
# filenames, such as:
# Tp_D_NRN_M_B_nat10134_nat00029_01004.tif

def get_source_ids(filename):
    parts = Path(str(filename)).stem.split("_")

    if len(parts) < 8 or parts[0].lower() != "tp":
        return None

    return parts[-3].lower(), parts[-2].lower()


# Build connected groups: images sharing either source identifier
# belong to the same group.
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

records = []
unparsed = []

for _, row in df.iterrows():
    filename = row["corrected_filename"]
    sources = get_source_ids(filename)

    if sources is None:
        unparsed.append(filename)
        continue

    a, b = sources
    union(a, b)
    records.append({
        "filename": filename,
        "source_1": a,
        "source_2": b,
    })

# Resolve connected components after all unions are complete.
for record in records:
    record["group_id"] = find(record["source_1"])

group_counts = Counter(r["group_id"] for r in records)

print("\n===== SOURCE-GROUP INSPECTION =====")
print("Manifest rows:", len(df))
print("Parsed filenames:", len(records))
print("Unparsed filenames:", len(unparsed))
print("Distinct source IDs:", len(parent))
print("Independent source groups:", len(group_counts))
print("Group sizes:", sorted(group_counts.values(), reverse=True))

print("\nLargest groups:")
for group, count in group_counts.most_common(15):
    print(f"{group}: {count} records")

if unparsed:
    print("\nExample filenames needing inspection:")
    for name in unparsed[:10]:
        print(name)

print("\nNo files were changed.")
