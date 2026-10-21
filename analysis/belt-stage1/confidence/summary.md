# Confidence report

Model: `models/resnext50_metal_plastic.pt`
Data: `data_belt_stage1`

## Validation

Images: 1251.
Argmax errors after scaling: 5.
Temperature: 1.011839 (fitted on validation NLL).
Validation NLL: 0.016391 raw, 0.016389 scaled.

Wrong argmax calls, lowest scaled score first:

- capture_00208_20260917_123800_024132_OTHERS.jpg | true metal | pred plastic | raw 76.46% | scaled 76.21%
- capture_00324_20260917_124502_946802_OTHERS.jpg | true metal | pred plastic | raw 84.87% | scaled 84.61%
- capture_00132_20260914_101313_047113_PP.jpg | true plastic | pred metal | raw 91.10% | scaled 90.88%
- capture_01024_20260916_130929_398443_HDPE.jpg | true plastic | pred metal | raw 96.25% | scaled 96.11%
- capture_00551_20260915_151544_271089_HDPE.jpg | true plastic | pred metal | raw 99.11% | scaled 99.06%

Threshold rule: among scores from 0.50 to 0.99 that hold at most 5% of validation images, pick the one that catches the most wrong argmax calls. Ties take the lower threshold. If none catch an error, leave sorting two-class.

Chosen threshold: 0.97. Holds 39 images (3.12%), catches 4 of 5 errors. Balanced accuracy on the rest: 99.95%.

Sweep rows that catch a new error, plus the choice:

- 0.50: held 0 (0.00%), caught 0, balanced accuracy 99.11%
- 0.77: held 6 (0.48%), caught 1, balanced accuracy 99.48%
- 0.85: held 14 (1.12%), caught 2, balanced accuracy 99.86%
- 0.91: held 21 (1.68%), caught 3, balanced accuracy 99.91%
- 0.97: held 39 (3.12%), caught 4, balanced accuracy 99.95% chosen

Full sweep: `threshold_sweep.csv`.
Histogram: `confidence_hist_validation.png`.

## Test

Scored once after the temperature and threshold above were fixed.
Images: 300.

Threshold 0.97 on the scaled winning score: 285 accepted, 15 held (5.00% of 300).
Accuracy on accepted images: 100.00% (285/285).
Balanced accuracy on accepted images: 100.00%.
metal recall on accepted: 142/142 (100.00%).
plastic recall on accepted: 143/143 (100.00%).
Held and wrong: 1. Held and correct: 14.

Crushed can `capture_01896_20260929_141207_925981.jpg`: true metal, pred plastic, raw 69.16%, scaled 68.96%, held.

Top-label ECE, 10 equal-mass bins: raw 0.0036, scaled 0.0037.
P(metal) ECE, 10 equal-mass bins: raw 0.0082, scaled 0.0083.
Brier score: raw 0.004700, scaled 0.004728.
Reliability diagram: `reliability_test.png`.

Burst agreement uses argmax, not the hold. Groups are timestamped frames in the same class folder. A frame with no timestamp is left out. A group of one frame is left out.

2s: 17/18 groups agree (94.44%), largest group 28, 100 frames without a timestamp.
5s: 9/10 groups agree (90.00%), largest group 28, 100 frames without a timestamp.

Augmentation: 5 passes of the training crop, flip, and ±15° rotation, seed 0. A pass flips when its argmax differs from the full-frame argmax.

metal: 10/750 passes flipped (1.33%), 4/150 images flipped at least once (2.67%).
plastic: 9/750 passes flipped (1.20%), 6/150 images flipped at least once (4.00%).
