# CASIA2 Image Forgery Detection and Segmentation

## 1. Project Overview

This project develops an image forgery localization and segmentation pipeline using the CASIA2 dataset.

The primary objective is to train a deep learning model to identify manipulated regions in tampered images using corresponding ground-truth masks.

The current baseline model is a U-Net segmentation network implemented with PyTorch.

---

## 2. Dataset

### Verified dataset

The verified working dataset contains:

- 141 tampered images
- 141 corresponding ground-truth masks
- 141 visually verified image-mask pairs

The ground-truth masks are binary masks:

- `0` = background / non-manipulated region
- `255` = manipulated region

Mask inspection confirmed that all 141 masks are binary.

### Mask statistics

Total inspected masks:

```text
141
```

Total pixels:

```text
Background (0):  18,387,157
Manipulated (255): 2,764,955
```

Mean manipulated-pixel ratio:

```text
0.120647
```

This indicates that the manipulated regions occupy only a portion of the image area, creating a significant class imbalance between background and manipulated pixels.

---

## 3. Image-Mask Verification

The image-mask matching process was performed using the corrected CASIA2 filenames and ground-truth mask naming structure.

Verification included:

- Filename matching
- Image existence checking
- Ground-truth mask existence checking
- Binary mask validation
- Visual verification of manipulated regions
- Verification that red highlighted regions correspond to manipulated areas

Final verification:

```text
Verified image-mask pairs: 141
Images present:             141
Masks present:              141
Review status:              VISUALLY_VERIFIED
```

The original dataset files were not modified.

---

## 4. Dataset Splitting

The verified dataset was divided into training, validation, and test sets.

| Split | Number of pairs |
|---|---:|
| Training | 98 |
| Validation | 21 |
| Test | 22 |
| Total | 141 |

The split was checked for:

- Missing images
- Missing masks
- Duplicate image paths
- Duplicate mask paths
- Shared image paths
- Shared mask paths
- Shared source groups

### Split validation result

```text
Train:
Rows: 98
Missing images: 0
Missing masks: 0
Duplicate image paths: 0
Duplicate mask paths: 0

Validation:
Rows: 21
Missing images: 0
Missing masks: 0
Duplicate image paths: 0
Duplicate mask paths: 0

Test:
Rows: 22
Missing images: 0
Missing masks: 0
Duplicate image paths: 0
Duplicate mask paths: 0
```

Cross-split image and mask overlap:

```text
Train vs Validation: 0
Train vs Test:       0
Validation vs Test:  0
```

Total unique image paths:

```text
141
```

Total unique mask paths:

```text
141
```

---

## 5. Project Structure

The important project structure is:

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
├── inspect_validation_predictions.py
├── predict_unet.py
├── evaluate_unet.py
├── evaluate_unet_test.py
│
├── README.md
│
└── prepared/
    │
    ├── correction_matching_report_v2.csv
    ├── unresolved_rows.csv
    ├── mask_label_inspection.csv
    │
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
        ├── validation_all_predictions/
        └── test_predictions/
```

---

## 6. Software Environment

The baseline implementation uses:

```text
Python
PyTorch
NumPy
Pandas
Pillow
Matplotlib
scikit-learn
```

The current PyTorch environment reports:

```text
PyTorch: 2.15.0+cpu
CUDA available: False
```

Therefore, the current U-Net experiment was trained using the CPU.

---

## 7. U-Net Baseline

The baseline model is a lightweight U-Net designed for pixel-level forgery localization.

The architecture contains:

- Four encoder blocks
- Max-pooling layers
- A bottleneck layer
- Four decoder blocks
- Skip connections
- Transposed convolution upsampling
- One-channel output for binary segmentation

Input:

```text
RGB image
```

Output:

```text
Binary manipulated-region mask
```

Input training resolution:

```text
256 × 256
```

The output prediction uses a sigmoid probability followed by a threshold of:

```text
0.5
```

---

## 8. Training Configuration

Training configuration:

```text
Training pairs:     98
Validation pairs:   21
Epochs:             30
Batch size:         2
Learning rate:      0.001
Optimizer:          Adam
Input resolution:   256 × 256
Device:             CPU
```

The training loss combines:

```text
Binary Cross-Entropy
+
Dice Loss
```

The model checkpoint is selected according to validation Dice score.

---

## 9. Baseline Training Result

The U-Net training completed successfully for 30 epochs.

Best validation Dice:

```text
0.5835
```

Training completed without a runtime error.

The best model was saved as:

```text
prepared/unet_baseline/best_unet.pth
```

Training history was saved as:

```text
prepared/unet_baseline/training_history.csv
```

---

## 10. Validation Evaluation

The trained model was evaluated independently across the 21 validation images.

Current validation results:

| Metric | Result |
|---|---:|
| Mean Dice | 0.2545 |
| Mean IoU | 0.2121 |
| Mean Precision | 0.2595 |
| Mean Recall | 0.2647 |
| Mean Accuracy | 0.8852 |
| Best Dice | 0.9636 |
| Worst Dice | 0.0000 |

### Interpretation

The validation performance is highly variable.

Some images produce strong segmentation:

```text
Dice = 0.9636
Dice = 0.9554
Dice = 0.8585
Dice = 0.8158
```

However, several validation images have a Dice score of:

```text
0.0000
```

Therefore, the current U-Net should be treated as a preliminary baseline rather than a final model.

Pixel accuracy is not considered the primary metric because the manipulated regions occupy a relatively small portion of the image. A model that predicts mostly background can obtain high pixel accuracy without providing useful forgery localization.

Dice and IoU are therefore emphasized.

---

## 11. Validation Prediction Inspection

Prediction visualization is performed using:

```text
inspect_validation_predictions.py
```

The visualization compares:

```text
Original image
Ground-truth mask
U-Net prediction
Prediction overlay
```

Prediction previews are stored in:

```text
prepared/unet_baseline/validation_all_predictions/
```

The inspection is used to determine why some validation examples receive very low Dice scores.

Possible failure modes include:

- Predicting mostly background
- Missing small manipulated regions
- Incorrect localization
- Excessive false-positive regions
- Loss of fine manipulation boundaries
- Insufficient resolution
- Class imbalance

---

## 12. Test Set

The test set contains:

```text
22 verified image-mask pairs
```

The test set must remain untouched during model development and hyperparameter tuning.

The correct experimental procedure is:

```text
Training set
    ↓
Model training
    ↓
Validation set
    ↓
Model selection / improvement
    ↓
Final model
    ↓
Test set
    ↓
Final reported performance
```

The test set should not be repeatedly used to select model settings.

---

## 13. Planned Test Metrics

The final U-Net will be evaluated using:

### Dice coefficient

Measures overlap between predicted and ground-truth manipulated regions.

```text
Dice = 2TP / (2TP + FP + FN)
```

### Intersection over Union

Measures the overlap relative to the union of prediction and ground truth.

```text
IoU = TP / (TP + FP + FN)
```

### Precision

Measures the proportion of predicted manipulated pixels that are correct.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures the proportion of actual manipulated pixels detected by the model.

```text
Recall = TP / (TP + FN)
```

### F1-score

Provides a harmonic mean of precision and recall.

---

## 14. Current Status

```text
[✓] CASIA2 dataset prepared
[✓] Filename corrections processed
[✓] Image-mask matching completed
[✓] 141 image-mask pairs verified
[✓] Binary masks verified
[✓] Visual verification completed
[✓] Train/validation/test split created
[✓] Split integrity checked
[✓] U-Net baseline implemented
[✓] U-Net baseline trained
[✓] Validation prediction previews generated
[✓] Validation metrics calculated
[ ] Validation failure analysis
[ ] U-Net improvement
[ ] Final model selection
[ ] Test-set evaluation
[ ] Final test visualization
[ ] Final result comparison
```

---

## 15. Important Experimental Rule

The 22-image test set should not be used for:

- Hyperparameter tuning
- Architecture selection
- Threshold selection
- Training
- Data augmentation decisions
- Selecting the best model

All model improvements should be evaluated using the training and validation sets.

Only the final selected model should be evaluated on the test set.

---

## 16. Planned Improvements

After analyzing the validation failures, possible improvements include:

### Improvement 1 — Data augmentation

Possible transformations:

- Horizontal flipping
- Vertical flipping
- Small rotations
- Cropping
- Brightness variation
- Contrast variation

Image and mask transformations must remain spatially synchronized.

### Improvement 2 — Higher input resolution

The current baseline uses:

```text
256 × 256
```

A higher resolution may preserve smaller manipulation regions and boundaries.

Possible future resolutions:

```text
384 × 384
512 × 512
```

depending on computational resources.

### Improvement 3 — Class-imbalance handling

The manipulated regions occupy a relatively small percentage of pixels.

Possible approaches:

- Weighted BCE
- Dice loss
- BCE + Dice
- Focal loss

### Improvement 4 — Transfer learning

A pretrained encoder can potentially provide stronger image representations than training the entire encoder from scratch.

---

## 17. Reproducibility

Random seed:

```text
42
```

The dataset manifests should be preserved together with the trained model.

Important files for reproducing the baseline:

```text
prepared/dataset_split/train_manifest.csv
prepared/dataset_split/validation_manifest.csv
prepared/dataset_split/test_manifest.csv
prepared/unet_baseline/best_unet.pth
prepared/unet_baseline/training_history.csv
```

---

## 18. Important Limitation

The current verified dataset contains 141 tampered image-mask pairs.

Therefore, the current experiment evaluates **tampered-region segmentation**, rather than providing a complete binary image-level forgery detector that distinguishes authentic images from manipulated images.

For a stronger image-forgery detection study, authentic images and appropriate all-background masks should also be incorporated into the experimental design.

Results obtained from this small verified subset should therefore be interpreted as a baseline experiment.

---

## 19. Final Research Goal

The final objective is to develop and evaluate a deep-learning-based forgery localization system capable of:

1. Receiving an image.
2. Detecting manipulated regions.
3. Producing a pixel-level forgery mask.
4. Comparing the predicted mask with ground truth.
5. Reporting Dice, IoU, precision, and recall.
6. Providing visual evidence of localization quality.
7. Comparing the baseline U-Net against improved segmentation approaches.

---

## 20. Current Baseline Summary

```text
Dataset:
CASIA2 verified tampered subset

Verified pairs:
141

Train:
98

Validation:
21

Test:
22

Model:
U-Net

Input:
256 × 256 RGB

Training:
30 epochs

Device:
CPU

Best validation Dice:
0.5835

Mean validation Dice:
0.2545

Mean validation IoU:
0.2121

Mean validation Precision:
0.2595

Mean validation Recall:
0.2647
```

The next stage is to analyze the validation predictions, improve the segmentation model using only the training/validation data, select the final model, and then perform one final evaluation on the 22-image test set.