# Metal / plastic classifier — training record

**Model.** ResNeXt-50 (32×4d), two classes.  
**Checkpoint.** `models/resnext50_metal_plastic_belt_stage1_layer4.pt` (also `models/named/belt_stage1_layer4.pt` and the default `models/resnext50_metal_plastic.pt`).  
**Trained.** 5 October 2026, Google Colab T4. **Record.** 16 October 2026.

This file is the record of the current belt model: what it was trained on, how it was trained, how it scored the locked test split, and the hold rule measured after that. The formatted copy of the training and test pages is `docs/belt_stage1_model_report.pdf`. That PDF does not include the uncertain-frame section.

## Dataset

10,806 images at 224×224. Train, validation, and test share neither filename nor file bytes. The test split is balanced on purpose (150 metal, 150 plastic).


| Split      | Stage 1 metal | Belt metal | Stage 1 plastic | Belt plastic | Total      |
| ---------- | ------------- | ---------- | --------------- | ------------ | ---------- |
| Train      | 680           | 680        | 3,949           | 3,946        | 9,255      |
| Validation | 40            | 93         | 406             | 712          | 1,251      |
| Test       | 40            | 110        | 60              | 90           | 300        |
| **Total**  | **760**       | **883**    | **4,415**       | **4,748**    | **10,806** |


Plastic is PET, HDPE, and PP. Metal is the non-plastic bucket: belt cans plus stage 1 photos labeled OTHERS (foil trays and other rejects). A photo in that folder is not necessarily made of metal.

Belt photos run from 11 September through 30 September 2026. The 28–30 September cans are held out in contiguous time windows. A two-frame gap on each side of a window is left out, so a near-duplicate does not land in two splits. Train is about 5.8 plastic images per metal image.

## Training procedure

The backbone is ResNeXt-50 (32×4d). Its 1,000-way ImageNet head is replaced by a linear layer from 2,048 features to 2 classes. This checkpoint was fine-tuned in two stages on the combined set above. Both stages resumed the previous weights. The run started from the pinned Drive head-only checkpoint (validation 99.07%, test 98.87% on the earlier Drive set), not from a fresh ImageNet initialization.

Stage 1 froze every residual stage and trained only the classification head, for 6 epochs at learning rate 0.0003. The best epoch reached 95.61% balanced validation accuracy (metal 123/133, plastic 1,104/1,118) and 96.67% balanced test accuracy (142/150 metal, 148/150 plastic).

Stage 2 unfroze the last residual stage, `layer4`, and left `layer1` through `layer3` frozen. It ran 4 epochs. The head stayed at learning rate 0.0003; `layer4` used 0.00003. The saved weights are the epoch with the highest balanced validation accuracy, the unweighted mean of metal recall and plastic recall. That was epoch 3, at 99.11% (metal 131/133, plastic 1,115/1,118). Epoch 4 matched those two counts and was not kept, because the saver replaces the file only when the balanced score rises.

## Training parameters


| Parameter          | Value                                                                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Optimizer          | AdamW, weight decay 0.01 (PyTorch default). No learning-rate schedule.                                                                          |
| Loss               | Cross-entropy. Training weights are inverse class frequency raised to 0.5: metal 1.845, plastic 0.766. Validation and test loss are unweighted. |
| Batch size         | 32                                                                                                                                              |
| Input              | 224×224, then ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)                                                                 |
| Train augmentation | Random resized crop, scale 0.6–1.0 and aspect 0.9–1.1; horizontal flip; rotation ±15°                                                           |
| Evaluation         | Full frame, bilinear resize to 224×224, then the same normalization                                                                             |
| Checkpoint rule    | Highest balanced validation accuracy                                                                                                            |
| Device             | NVIDIA T4                                                                                                                                       |




## Test results

The 300-image test split was scored once, after the checkpoint had been chosen on validation. It was not used to pick the epoch.


|                   | Metal  | Plastic | Overall      |
| ----------------- | ------ | ------- | ------------ |
| Images            | 150    | 150     | 300          |
| Correct           | 149    | 150     | 299          |
| Recall            | 99.33% | 100%    | —            |
| Precision         | 100%   | 99.34%  | —            |
| F1                | 99.67% | 99.67%  | 99.67% macro |
| Accuracy          | —      | —       | 99.67%       |
| Balanced accuracy | —      | —       | 99.67%       |


Confusion matrix. Rows are the true class. Columns are the prediction.


|              | Predicted metal | Predicted plastic |
| ------------ | --------------- | ----------------- |
| True metal   | 149             | 1                 |
| True plastic | 0               | 150               |


The one error is a crushed can from 29 September 2026, `capture_01896_20260929_141207_925981.jpg`, called plastic at 69%. A frame from the same burst, four seconds earlier, was classified correctly.

Validation, used only to choose the epoch, was 1,246/1,251 (99.60% accuracy, 99.11% balanced). The five misses are three plastic images called metal (two HDPE, one PP) and two metal images called plastic (both OTHERS).

## Uncertain frames

The checkpoint stays two classes. `uncertain` is a hold after the softmax: when the winning score is below 0.85, the frame is not sorted as metal or plastic. It is not a third training class, and it is not the stage 1 `OTHERS` resin folder (those photos were trained as metal). `predict.py` applies this rule when it loads the belt stage 1 checkpoint. `--threshold` overrides the cutoff. Other checkpoints keep the raw softmax unless a threshold is passed.

The score is the softmax of the logits divided by 1.011839. That temperature was fit on the 1,251 validation images by minimizing unweighted negative log likelihood. Validation NLL moved from 0.016391 to 0.016389. The probabilities barely change.

The threshold was chosen on those validation scores, and the test split was not used to pick it. The rule: among cutoffs from 0.50 to 0.99 that hold at most 5% of validation images, take the one that catches the most wrong argmax calls. Ties take the lower cutoff. The five validation misses, lowest scaled score first:


| Image | True | Predicted | Raw | Scaled |
| --- | --- | --- | --- | --- |
| `capture_00208_20260917_123800_024132_OTHERS.jpg` | metal | plastic | 76.46% | 76.21% |
| `capture_00324_20260917_124502_946802_OTHERS.jpg` | metal | plastic | 84.87% | 84.61% |
| `capture_00132_20260914_101313_047113_PP.jpg` | plastic | metal | 91.10% | 90.88% |
| `capture_01024_20260916_130929_398443_HDPE.jpg` | plastic | metal | 96.25% | 96.11% |
| `capture_00551_20260915_151544_271089_HDPE.jpg` | plastic | metal | 99.11% | 99.06% |


Cutoffs where a new validation error is caught. With no hold, balanced validation accuracy is 99.11%.


| Threshold | Held | Errors caught | Balanced accuracy on the rest |
| --- | --- | --- | --- |
| 0.77 | 6 (0.48%) | 1 of 5 | 99.48% |
| 0.85 | 14 (1.12%) | 2 of 5 | 99.86% |
| 0.91 | 21 (1.68%) | 3 of 5 | 99.91% |
| **0.97** | **39 (3.12%)** | **4 of 5** | **99.95%** |


0.97 is the cutoff that measurement selected. The HDPE bottle at 99.06% metal stays above every cutoff that holds at most 5% of validation.

The live cutoff in `decide.py` is 0.85. This checkpoint is a public showcase, so a few confident mistakes are sorted instead of parking the frame. On the same validation scores, 0.85 holds 14 images (1.12%) and catches 2 of 5 errors. The three that stay sorted are the PP bottle at 90.88% and the HDPE bottles at 96.11% and 99.06%. The same already-scored test logits hold 4 of 300 at 0.85, and the other 296 are correct. The crushed can at 68.96% is one of those four.

The 300-image test split was scored once with those two numbers already fixed. 15 images were held (5.00%). The other 285 were all correct: 142/142 metal and 143/143 plastic. Of the 15 held, 1 was the wrong argmax and 14 were correct. The crushed can `capture_01896_20260929_141207_925981.jpg` is the held error (raw 69.16%, scaled 68.96% plastic).

Top-label expected calibration error on that test pass, 10 equal-mass bins, is 0.0036 raw and 0.0037 scaled. The same bins on P(metal) are 0.0082 raw and 0.0083 scaled. Brier score is 0.004700 raw and 0.004728 scaled. Almost every winning score already sits near 1, so these gaps stay small. The histogram, the full sweep, and the reliability diagram are in `analysis/belt-stage1/confidence/`.

Burst agreement is the share of timestamped groups, in one class folder, whose argmax matches on every frame. A frame with no timestamp is left out (100 of 300). A group of one frame is left out. Within 2 seconds there are 18 groups, all metal, and 17 agree. Within 5 seconds there are 10 groups and 9 agree. The group that disagrees is the 28-frame burst from 29 September 2026 that contains the crushed can; the frames before and after it are called metal.

Five passes of the training crop, flip, and ±15° rotation, seed 0, were compared with the full-frame argmax. Metal flipped on 10/750 passes (4/150 images). Plastic flipped on 9/750 passes (6/150 images).

Plots for this checkpoint are in `analysis/belt-stage1/`.