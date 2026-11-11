# Knowledge Transfer — CASIA2 Forgery Segmentation Project

## 1. Project Purpose

This document provides the knowledge-transfer information required to understand, reproduce, continue, and extend the CASIA2 image forgery localization and segmentation project.

The current project uses verified tampered images and corresponding ground-truth masks to establish a U-Net segmentation baseline.

The system is designed to identify manipulated regions at the pixel level.

---

## 2. Current Project Objective

The primary objective is:

> Develop a deep-learning model that receives a potentially manipulated image and produces a pixel-level mask identifying the manipulated regions.

The current baseline architecture is U-Net.

The primary evaluation metrics are:

- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall

Pixel accuracy is retained as a secondary metric because the dataset contains substantial background regions.

---

## 3. Dataset Status

The current verified dataset contains:

```text
Total verified pairs: 141
```

Each pair consists of:

```text
Tampered image
+
Corresponding ground-truth manipulation mask
```

The masks have been visually checked to ensure that the highlighted regions correspond to the manipulated areas.

### Mask convention

```text
Pixel value 0   → Background / non-manipulated
Pixel value 255 → Manipulated region
```

All 141 inspected masks were classified as:

```text
BINARY_MASK
```

---

## 4. Dataset Split

The verified dataset has already been divided into three groups:

```text
Training     = 98 pairs
Validation   = 21 pairs
Testing      = 22 pairs
Total        = 141 pairs
```

The split was validated for:

- Missing images
- Missing masks
- Duplicate image paths
- Duplicate mask paths
- Cross-split image overlap
- Cross-split mask overlap
- Cross-split source-group overlap

The validation showed:

```text
Missing images: 0
Missing masks: 0
Duplicate image paths: 0
Duplicate mask paths: 0
Shared source groups across splits: 0
```

This split must be preserved during subsequent experiments.

---

# 5. Project Directory

The important directory structure is:

```text
CASIA2/
│
├── Au/
│   └── Authentic images
│
├── Tp/
│   └── Tampered images
│
├── CASIA 2 Groundtruth/
│   └── Ground-truth masks
│
├── fileNamesCorrection.xlsx
│
├── prepare_casia.py
├── check_corrections.py
├── inspect_mask_labels.py
├── predict_unet.py
├── evaluate_unet.py
├── inspect_validation_predictions.py
│
├── README.md
├── knowledge_transfer.md
│
└── prepared/
    │
    ├── correction_matching_report_v2.csv
    ├── unresolved_rows.csv
    ├── mask_label_inspection.csv
    ├── visual_verification_record.csv
    │
    ├── dataset_split/
    │   ├── train_manifest.csv
    │   ├── validation_manifest.csv
    │   └── test_manifest.csv
    │
    └── unet_baseline/
        ├── best_unet.pth
        ├── training_history.csv
        ├── validation_metrics.csv
        ├── validation_previews/
        └── validation_all_predictions/
```

---

# 6. Important Scripts

## `prepare_casia.py`

Purpose:

- Process the filename corrections
- Locate image files
- Locate corresponding masks
- Generate matching reports
- Prepare the dataset manifests

Important output:

```text
prepared/correction_matching_report_v2.csv
```

---

## `check_corrections.py`

Purpose:

- Investigate unresolved filename corrections
- Search the ground-truth directory
- Identify possible mask candidates
- Verify filename relationships

This script should not modify the original dataset.

---

## `inspect_mask_labels.py`

Purpose:

- Inspect ground-truth mask pixel values
- Confirm binary mask structure
- Calculate pixel statistics

Expected mask labels:

```text
0
255
```

---

## `predict_unet.py`

Purpose:

- Load the trained U-Net checkpoint
- Run predictions on validation images
- Generate visual comparisons

Generated output:

```text
prepared/unet_baseline/validation_previews/
```

---

## `evaluate_unet.py`

Purpose:

- Evaluate U-Net predictions on the validation set
- Calculate segmentation metrics

Current validation result:

```text
Mean Dice      = 0.2545
Mean IoU       = 0.2121
Mean Precision = 0.2595
Mean Recall    = 0.2647
Mean Accuracy  = 0.8852
```

---

## `inspect_validation_predictions.py`

Purpose:

- Generate visual inspection files for all validation images
- Compare original images, ground-truth masks, and U-Net predictions
- Investigate failure cases

The output is stored in:

```text
prepared/unet_baseline/validation_all_predictions/
```

---

# 7. Baseline Model

The current baseline is a lightweight U-Net.

### Encoder

```text
3 → 16
16 → 32
32 → 64
64 → 128
```

### Bottleneck

```text
128 → 256
```

### Decoder

```text
256 → 128
128 → 64
64 → 32
32 → 16
```

### Output

```text
16 → 1
```

The output represents the probability that each pixel belongs to a manipulated region.

---

# 8. Training Configuration

Current baseline configuration:

```text
Model:          U-Net
Input channels: 3
Output channels: 1
Image size:     256 × 256
Epochs:         30
Batch size:     2
Learning rate:  0.001
Optimizer:      Adam
Device:         CPU
```

The current environment reports:

```text
PyTorch: 2.15.0+cpu
CUDA available: False
```

Therefore, the current experiment was performed using CPU computation.

---

# 9. Loss Function

The training objective combines:

```text
Binary Cross-Entropy Loss
+
Dice Loss
```

The purpose of combining these losses is to provide both:

- Pixel-level classification learning
- Region-overlap optimization

This is particularly important because manipulated pixels represent a relatively small fraction of the total image area.

---

# 10. Baseline Training Result

The best validation Dice obtained during training was:

```text
0.5835
```

The best checkpoint is stored at:

```text
prepared/unet_baseline/best_unet.pth
```

This checkpoint should be preserved as the baseline reference model.

---

# 11. Validation Evaluation

The independently calculated validation results were:

```text
Mean Dice:
0.2545

Mean IoU:
0.2121

Mean Precision:
0.2595

Mean Recall:
0.2647

Mean Accuracy:
0.8852
```

The validation results are highly variable.

Examples of strong predictions:

```text
Dice = 0.9636
Dice = 0.9554
Dice = 0.8585
Dice = 0.8158
```

Several validation samples received:

```text
Dice = 0.0000
```

This indicates that the current baseline requires further investigation and improvement.

---

# 12. Important Difference Between Training and Evaluation Results

The following two numbers should not be confused:

```text
Best validation Dice during training = 0.5835
Mean Dice across validation evaluation = 0.2545
```

The first represents the best checkpoint-selection value recorded during training.

The second is the mean Dice obtained when evaluating the selected checkpoint across the 21 validation images.

For reporting, retain the distinction between:

- Checkpoint-selection metric
- Full validation-set evaluation metric

---

# 13. Why Accuracy Is Not the Main Metric

The dataset contains large background areas.

For example:

```text
Background = 18,387,157 pixels
Manipulated = 2,764,955 pixels
```

Therefore, a model can achieve relatively high pixel accuracy while failing to identify manipulated regions correctly.

For this reason, the primary metrics should be:

```text
1. Dice
2. IoU
3. Precision
4. Recall
```

Accuracy should be treated as supplementary.

---

# 14. Current Research Interpretation

The current U-Net should be considered a:

> Preliminary baseline for pixel-level image forgery localization.

The current results do not yet justify claiming that the model performs well on CASIA2 forgery segmentation.

The large variation between validation examples indicates that further investigation is required.

---

# 15. Validation Failure Analysis

Before changing the architecture, inspect the generated validation predictions.

Use:

```text
inspect_validation_predictions.py
```

Inspect especially the images with:

```text
Dice ≈ 0
```

Compare:

```text
Original image
        ↓
Ground-truth mask
        ↓
U-Net prediction
        ↓
Prediction overlay
```

Determine whether the failure is caused by:

- Complete background prediction
- Wrong manipulated-region location
- Very small manipulated regions
- Boundary errors
- False-positive regions
- Insufficient image resolution
- Class imbalance

The reason for failure should be identified before selecting the next model modification.

---

# 16. Recommended Improvement Sequence

Do not immediately change multiple variables simultaneously.

Use controlled experiments.

## Experiment A — Baseline

Keep the current configuration unchanged.

Record:

```text
Dice
IoU
Precision
Recall
```

This is the reference baseline.

---

## Experiment B — Data Augmentation

Introduce synchronized image-mask augmentation.

Possible transformations:

```text
Horizontal flip
Vertical flip
Small rotations
Brightness adjustment
Contrast adjustment
```

The same spatial transformation must be applied to both:

```text
Image
Mask
```

Do not independently transform the mask.

---

## Experiment C — Class Imbalance

Experiment with loss functions such as:

```text
BCE + Dice
Weighted BCE + Dice
Focal + Dice
```

Compare the validation metrics.

---

## Experiment D — Higher Resolution

The baseline currently uses:

```text
256 × 256
```

Possible alternatives:

```text
384 × 384
512 × 512
```

Higher resolution may improve detection of small manipulation regions but increases CPU computation and memory requirements.

---

## Experiment E — Improved Encoder

After establishing the basic U-Net results, consider a stronger encoder or transfer-learning-based segmentation architecture.

Possible future architectures:

```text
U-Net with pretrained encoder
U-Net++
Attention U-Net
DeepLabV3+
```

These should be introduced only after establishing the baseline and documenting the improvement.

---

# 17. Experimental Discipline

Only one major change should be introduced at a time whenever possible.

For example:

```text
Baseline U-Net
       ↓
Baseline + augmentation
       ↓
Baseline + improved loss
       ↓
Baseline + augmentation + improved loss
```

Record each experiment separately.

Do not overwrite the baseline checkpoint.

Use names such as:

```text
best_unet_baseline.pth
best_unet_augmentation.pth
best_unet_dice_loss.pth
best_unet_augmented_loss.pth
```

---

# 18. Test Set Protection

The test set contains:

```text
22 pairs
```

It must remain untouched during model development.

Do not use the test set for:

- Hyperparameter tuning
- Threshold tuning
- Architecture selection
- Loss selection
- Augmentation selection
- Model selection

The correct sequence is:

```text
TRAIN
  ↓
Train models
  ↓
VALIDATION
  ↓
Select final model
  ↓
LOCK MODEL
  ↓
TEST
  ↓
Final reported metrics
```

---

# 19. Final Test Evaluation

After the final model has been selected using the validation set, evaluate it once on:

```text
22 test image-mask pairs
```

The final report should contain:

```text
Dice
IoU
Precision
Recall
Accuracy
```

Also generate visual results:

```text
Original
Ground Truth
Prediction
Overlay
```

The test results should be reported separately from validation results.

---

# 20. Reproducibility Requirements

The following files should be retained:

```text
fileNamesCorrection.xlsx

prepared/
├── correction_matching_report_v2.csv
├── mask_label_inspection.csv
├── visual_verification_record.csv
│
├── dataset_split/
│   ├── train_manifest.csv
│   ├── validation_manifest.csv
│   └── test_manifest.csv
│
└── unet_baseline/
    ├── best_unet.pth
    ├── training_history.csv
    └── validation_metrics.csv
```

Do not manually edit the manifests after validation unless the entire experiment is intentionally rebuilt.

---

# 21. Current Project Status

```text
[✓] Dataset prepared
[✓] Filename corrections processed
[✓] Image-mask matching completed
[✓] 141 pairs verified
[✓] Masks inspected
[✓] Masks visually verified
[✓] Dataset split
[✓] Split integrity verified
[✓] PyTorch environment verified
[✓] U-Net baseline trained
[✓] Validation predictions generated
[✓] Validation metrics calculated
[✓] Knowledge-transfer documentation started

[ ] Analyze validation failures
[ ] Improve U-Net baseline
[ ] Compare improved models
[ ] Select final model
[ ] Evaluate 22-image test set
[ ] Generate final visualizations
[ ] Prepare final experimental results
```

---

# 22. Recommended Next Action

The immediate next task is **validation failure analysis**.

Run:

```text
python inspect_validation_predictions.py
```

Then inspect the generated directory:

```text
prepared/unet_baseline/validation_all_predictions/
```

Pay particular attention to:

```text
Dice = 0.000
```

and compare them with:

```text
Dice > 0.80
```

This comparison will identify the main weakness of the current baseline.

Do not evaluate the 22 test images yet.

---

# 23. Final Experimental Workflow

The complete project workflow is:

```text
CASIA2 Dataset
      ↓
Filename Correction
      ↓
Image-Mask Matching
      ↓
Binary Mask Verification
      ↓
Visual Verification
      ↓
141 Verified Pairs
      ↓
Group-Aware Dataset Split
      ↓
98 Train / 21 Validation / 22 Test
      ↓
Baseline U-Net
      ↓
Training
      ↓
Validation Evaluation
      ↓
Failure Analysis
      ↓
Model Improvements
      ↓
Validation Comparison
      ↓
Final Model Selection
      ↓
Model Locked
      ↓
22-Image Test Evaluation
      ↓
Final Metrics
      ↓
Final Visualizations
      ↓
Research Report
```

---

# 24. Key Handover Information

A new researcher continuing this project should know the following:

1. The working verified dataset contains 141 tampered image-mask pairs.
2. The ground-truth masks are binary.
3. The manipulated regions correspond to the white/255 regions of the masks.
4. The dataset has already been split into 98 training, 21 validation, and 22 test pairs.
5. The splits have been checked for overlap and missing files.
6. The baseline model is a U-Net.
7. The baseline was trained on CPU.
8. The best validation Dice recorded during training was 0.5835.
9. Full validation evaluation produced a mean Dice of 0.2545.
10. The validation results are highly variable.
11. The test set must remain untouched until final model selection.
12. Future improvements must be evaluated using the validation set first.
13. The baseline checkpoint must be preserved for comparison.
14. Final test evaluation should be performed only after the final model is locked.

---

# 25. Research Reporting Principle

Every model improvement should be supported by quantitative comparison.

A recommended final comparison table is:

| Model | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| U-Net Baseline | — | — | — | — |
| U-Net + Augmentation | — | — | — | — |
| U-Net + Improved Loss | — | — | — | — |
| Final Model | — | — | — | — |

Only experimentally measured values should be entered into this table.

Do not fabricate or estimate missing results.

---

## End of Knowledge Transfer Document