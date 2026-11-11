
from pathlib import Path
import html
import json
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
REPORT = BASE / "prepared" / "correction_matching_report_v2.csv"
OUT = BASE / "prepared" / "visual_review"
ASSETS = OUT / "assets"

OUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

if not REPORT.exists():
    raise FileNotFoundError(f"Report not found: {REPORT}")

df = pd.read_csv(REPORT)

required = {"original_filename", "corrected_filename",
            "image_path", "mask_path"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing CSV columns: {sorted(missing)}")

# Preserve existing human review decisions when rerunning.
review_csv = OUT / "visual_review_results.csv"

if review_csv.exists():
    old = pd.read_csv(review_csv)
    if {"original_filename", "review_decision"} <= set(old.columns):
        decisions = old.set_index("original_filename")[
            "review_decision"
        ].to_dict()
    else:
        decisions = {}
else:
    decisions = {}

rows = []
cards = []

def make_thumbnail(image, path, max_size=(480, 360)):
    image = ImageOps.exif_transpose(image)
    image.thumbnail(max_size)
    image.convert("RGB").save(path, quality=90)

for index, row in df.iterrows():
    original = str(row["original_filename"])
    corrected = str(row["corrected_filename"])
    image_path = Path(str(row["image_path"]))
    mask_path = Path(str(row["mask_path"]))

    # Reject missing paths and pandas' empty-cell representation.
    if not image_path.is_file() or not mask_path.is_file():
        rows.append({
            "original_filename": original,
            "corrected_filename": corrected,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_size": "",
            "mask_size": "",
            "dimensions_match": False,
            "mask_values": "",
            "review_decision": decisions.get(original, ""),
            "notes": "IMAGE OR MASK FILE MISSING"
        })
        continue

    try:
        with Image.open(image_path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
            image_size = image.size

        with Image.open(mask_path) as mm:
            mask = ImageOps.exif_transpose(mm).convert("L")
            mask_size = mask.size
            mask_array = np.array(mask)

        dimensions_match = image_size == mask_size
        values = np.unique(mask_array)
        values_text = ",".join(map(str, values[:20]))

        # A red overlay is a visual aid, not proof of correct matching.
        # Resize only for display if dimensions differ; original files
        # remain unchanged.
        display_mask = mask.resize(image_size, Image.Resampling.NEAREST)
        mask_display = np.array(display_mask)
        foreground = mask_display > 127

        image_array = np.array(image).copy()
        overlay = image_array.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 255

        overlay[foreground] = (
            0.55 * overlay[foreground] +
            0.45 * red[foreground]
        ).astype(np.uint8)

        stem = f"{index:04d}"
        image_thumb = ASSETS / f"{stem}_image.jpg"
        mask_thumb = ASSETS / f"{stem}_mask.jpg"
        overlay_thumb = ASSETS / f"{stem}_overlay.jpg"

        make_thumbnail(image, image_thumb)
        make_thumbnail(display_mask, mask_thumb)
        make_thumbnail(Image.fromarray(overlay), overlay_thumb)

        notes = []
        if not dimensions_match:
            notes.append("DIMENSION MISMATCH")
        if len(values) > 20:
            notes.append("Many mask intensity values: inspect mask encoding")
        if len(values) == 1:
            notes.append("Mask is constant: inspect")

        decision = decisions.get(original, "")

        rows.append({
            "original_filename": original,
            "corrected_filename": corrected,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_size": str(image_size),
            "mask_size": str(mask_size),
            "dimensions_match": dimensions_match,
            "mask_values": values_text,
            "review_decision": decision,
            "notes": "; ".join(notes)
        })

        def asset_link(path):
            return "assets/" + path.name

        status = "DIMENSION MISMATCH" if not dimensions_match else "Review needed"

        cards.append(f"""
        <section class="card">
          <h3>Pair {index + 1}: {html.escape(original)}</h3>
          <p><b>Corrected name:</b> {html.escape(corrected)}</p>
          <p><b>Status:</b> {html.escape(status)}</p>
          <div class="images">
            <figure><img src="{asset_link(image_thumb)}">
              <figcaption>Original image</figcaption></figure>
            <figure><img src="{asset_link(mask_thumb)}">
              <figcaption>Ground-truth mask</figcaption></figure>
            <figure><img src="{asset_link(overlay_thumb)}">
              <figcaption>Red overlay</figcaption></figure>
          </div>
          <p><b>Image size:</b> {image_size};
             <b>Mask size:</b> {mask_size}</p>
          <p><b>Mask values:</b> {html.escape(values_text)}</p>
          <label>Decision:
            <select data-name="{html.escape(original, quote=True)}">
              <option value="">Not reviewed</option>
              <option value="MATCH">Match confirmed</option>
              <option value="MISMATCH">Incorrect pair</option>
              <option value="UNCERTAIN">Uncertain</option>
            </select>
          </label>
          <label>Notes:
            <input class="note" data-name="{html.escape(original, quote=True)}"
                   placeholder="Optional review notes">
          </label>
        </section>
        """)

    except Exception as exc:
        rows.append({
            "original_filename": original,
            "corrected_filename": corrected,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_size": "",
            "mask_size": "",
            "dimensions_match": False,
            "mask_values": "",
            "review_decision": decisions.get(original, ""),
            "notes": f"READ ERROR: {exc}"
        })

results = pd.DataFrame(rows)
results.to_csv(review_csv, index=False)

# Browser gallery. Save decisions to a downloadable CSV.
page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>CASIA2 Image-Mask Review</title>
<style>
body {{font-family:Arial,sans-serif;margin:20px;background:#f4f4f4}}
.card {{background:white;padding:16px;margin:16px 0;border-radius:8px}}
.images {{display:flex;flex-wrap:wrap;gap:12px}}
figure {{margin:0;flex:1;min-width:220px}}
img {{width:100%;height:auto;border:1px solid #ccc}}
figcaption {{text-align:center;padding:6px}}
label {{display:inline-block;margin:8px 12px 8px 0}}
input.note {{width:260px}}
</style></head>
<body>
<h1>CASIA2 image and ground-truth review</h1>
<p>Review each pair visually. A matching filename or image size alone
does not establish that the mask belongs to the image.</p>
<button onclick="downloadResults()">Download review CSV</button>
<div id="cards">{''.join(cards)}</div>
<script>
const previous = {{}};
const originalDecisions = {json.dumps(decisions)};
for (const el of document.querySelectorAll("select[data-name]")) {{
  el.value = originalDecisions[el.dataset.name] || "";
}}
function downloadResults() {{
  const data = {json.dumps(results.to_dict(orient="records"))};
  const decisions = {{}};
  document.querySelectorAll("select[data-name]").forEach(el => {{
    decisions[el.dataset.name] = el.value;
  }});
  const notes = {{}};
  document.querySelectorAll("input.note[data-name]").forEach(el => {{
    notes[el.dataset.name] = el.value;
  }});
  const cols = Object.keys(data[0] || {{}});
  const quote = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
  const lines = [cols.map(quote).join(",")];
  data.forEach(row => {{
    row.review_decision = decisions[row.original_filename] || "";
    row.notes = notes[row.original_filename] || row.notes || "";
    lines.push(cols.map(c => quote(row[c])).join(","));
  }});
  const blob = new Blob([lines.join("\\r\\n")],
    {{type:"text/csv;charset=utf-8"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "visual_review_results.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}}
</script></body></html>"""

# json is used for safe JavaScript data serialization.
import json
page = page.replace(
    f"<script>\nconst previous = {{}};",
    f"<script>\nconst previous = {{}};"
)
(OUT / "index.html").write_text(page, encoding="utf-8")

print("Total pairs:", len(results))
print("Missing/read-error rows:",
      results["notes"].astype(str).str.contains(
          "MISSING|READ ERROR", regex=True
      ).sum())
print("Dimension mismatches:",
      (~results["dimensions_match"].fillna(False)).sum())
print("Gallery:", OUT / "index.html")
print("Review CSV:", review_csv)
print("Original images and masks were not modified.")
