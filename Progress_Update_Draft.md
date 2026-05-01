# Deep Learning for Wheat Detection in Aerial Images
**CS444/ECE494 Four-Credit Project Progress Update**

## 1. Group Members
- Gezhi Zou

## 2. Project Title
**Deep Learning for Wheat Detection in Aerial Images**

## 3. Updated Problem Definition and Plan
The project goal remains the same as in the proposal: build an object detection system that detects wheat heads in aerial field images and outputs bounding boxes with confidence scores.

### 3.1 What Changed From the Proposal (and Why)
The model family did not change (Faster R-CNN with transfer learning), but the implementation plan was refined based on real training/runtime behavior:

1. Baseline-first strategy was enforced.
- I implemented and stabilized a working `torchvision` Faster R-CNN baseline before trying broader architecture changes.
- Reason: this reduces project risk and creates a reliable reference point for future ablations.

2. Training loop was optimized for runtime.
- During development, per-epoch evaluation (validation loss + accuracy + inference-count metrics) added significant overhead.
- The latest code flow now prioritizes faster iteration: training is performed per epoch, and heavier evaluation routines are executed at the end of training.
- Reason: faster feedback cycles, lower GPU overhead, and easier repeated experiments.

3. Reproducibility and logging were strengthened.
- Fixed random seed and deterministic split logic are used.
- Training history is saved to CSV, and plots/checkpoints are saved to disk.
- Reason: experiment comparisons are clearer and easier to report.

### 3.2 Self-Contained Current Plan
Current end-to-end pipeline:
1. Parse Global Wheat Detection annotations (`train.csv`) into detection targets.
2. Build train/validation split from training image IDs.
3. Train Faster R-CNN (ResNet-FPN backbone) with transfer learning.
4. Save checkpoints (`best_model.pt`, `last_model.pt`) and training logs.
5. Produce qualitative visualizations:
- augmentation examples,
- target (ground-truth) output examples,
- prediction examples on validation data.
6. Run final evaluation and summarize strengths/failure cases.

## 4. Illustrations of Data and/or Problem
I added explicit notebook cells for all required visual evidence:

1. Data examples (raw/annotated)
- The notebook now shows example training images with ground-truth wheat boxes.

2. Data augmentation examples
- A dedicated section compares `augment=False` vs `augment=True` samples side-by-side for the same image IDs.
- Current augmentation used in code: random horizontal flip.

3. Target output examples
- A dedicated section prints a real example target dictionary fields:
- `boxes` shape and sample values,
- `labels`,
- `area`,
- `iscrowd`.

4. Model prediction examples
- A dedicated section visualizes validation samples with:
- left: ground-truth boxes,
- right: predicted boxes with confidence scores.

These were added directly in:
- `src/faster_rcnn_wheat.ipynb`

## 5. Key Resources
### 5.1 Dataset
- Global Wheat Detection (Kaggle):
  - <https://www.kaggle.com/competitions/global-wheat-detection>
- Used data:
  - `train.csv`
  - `data/train/*.jpg`
  - optional test images for qualitative checks

### 5.2 Code Bases and Libraries
- PyTorch: <https://pytorch.org>
- Torchvision detection models: <https://pytorch.org/vision/stable/models.html>
- Matplotlib + PIL for visualization

### 5.3 Computing Platform
- Local workstation with NVIDIA RTX 3080 Ti GPU
- Local Python environment for training and notebook experiments

### 5.4 Build vs Reuse Clarification
Implemented by me:
- dataset parsing and bbox conversion,
- train/val split logic,
- training loop and experiment logging,
- output plotting and checkpoint saving,
- augmentation demo + target-output demo + prediction demo notebook sections.

Reused existing implementations:
- base detector architecture from `torchvision.models.detection.fasterrcnn_resnet50_fpn`.

Pretraining status:
- transfer learning from pre-trained weights is used.

## 6. Summary of Work Done to Date
### 6.1 Completed Engineering Work
1. Built a full training script and notebook pipeline.
2. Verified dataset loading and annotation parsing.
3. Implemented Faster R-CNN baseline training and checkpointing.
4. Added training history export and plots.
5. Added qualitative prediction preview generation.
6. Added explicit augmentation and target-output demo cells in the notebook.
7. Added explicit prediction visualization examples in the notebook.
8. Refined runtime strategy to reduce unnecessary repeated evaluation overhead.

### 6.2 Evidence of Successfully Running Code
Baseline code has run successfully and produced artifacts in `outputs_notebook/`:
- `best_model.pt`
- `last_model.pt`
- `training_history.csv`
- `loss_curves.png`
- `epoch_runtime.png`
- `val_box_count_trend.png` (from earlier evaluation mode)
- `val_prediction_preview.png`

Representative baseline training results from `outputs_notebook/training_history.csv`:
- Epoch 1 train loss: **0.8976**
- Epoch 5 train loss: **0.6562**
- Validation average predicted boxes/image moved near validation average GT boxes/image (roughly low-40s range by epoch 5), indicating improved calibration of detection count behavior.

This confirms the pipeline trains end-to-end and produces measurable outputs.

### 6.3 Current Baseline Table
| Experiment | Model | Epochs | LR schedule | Key Result |
|---|---|---:|---|---|
| Baseline run | Faster R-CNN (ResNet-FPN, transfer learning) | 5 | StepLR | Train loss decreased from 0.8976 to 0.6562 |

## 7. Plan for the Remainder of the Project
### 7.1 Minimum Goal
1. Deliver one stable, reproducible baseline detector with:
- complete training logs,
- final quantitative metrics,
- qualitative success/failure visualization.
2. Provide ablation on at least one training setting (e.g., augmentation policy or LR).

### 7.2 Maximum Goal
1. Compare at least one stronger variant against the baseline (architecture or training strategy).
2. Improve dense-scene performance with targeted tuning (thresholds/augmentation/sampling).
3. Produce a clearer error taxonomy and mitigation analysis.

### 7.3 Risks and Mitigations
1. Runtime bottlenecks on heavy evaluation.
- Mitigation: evaluate expensive metrics at final stage (current code direction).

2. Overfitting or unstable generalization.
- Mitigation: controlled augmentation, fixed split, and explicit qualitative review.

3. Dense-object misses / localization errors.
- Mitigation: threshold tuning, more targeted qualitative analysis, and focused ablation.

## 8. Member Roles and Collaboration Plan
This is currently a one-member project.

- Gezhi Zou owns:
1. Data preprocessing and pipeline implementation
2. Model training and checkpointing
3. Experiment tracking and analysis
4. Visualization and report writing
5. Final packaging/presentation

Execution plan going forward:
1. Continue milestone-based iterations (baseline -> ablations -> final comparison).
2. Keep experiment settings/results synchronized with report text and figures.
3. Freeze a final reproducible run for submission evidence.

## 9. Updated AI Use Plan
(Kept unchanged from proposal, as requested.)

AI tools may be used in a limited capacity during this project. Specifically, AI will be used primarily to assist with:
- learning the usage of machine learning frameworks such as PyTorch, including understanding APIs and model training workflows (e.g., how to implement neural network layers or training loops)
- learning about deep learning architectures such as ResNet and other related models through explanations and examples
- generating or organizing report templates and improving the clarity of written documentation

AI tools will not be used to directly generate implementation code for the models in this project. All model implementations, data processing, training pipelines, and experiments will be written manually by the author.

The purpose of using AI in this project is mainly to support learning and documentation, rather than to automate coding tasks. All AI usage will be documented clearly in the final report as required by the course policy.

## 10. Reference List
1. Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* <https://arxiv.org/abs/1506.01497>
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* <https://arxiv.org/abs/1512.03385>
3. Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* <https://pytorch.org>
4. Kaggle. (2026). *Global Wheat Detection.* <https://www.kaggle.com/competitions/global-wheat-detection>

---

## Appendix: Figure Checklist For Final PDF/DOC
1. Example raw aerial image.
2. Same image with GT boxes.
3. Augmentation comparison panel (`augment=False` vs `augment=True`).
4. Target dictionary example screenshot (boxes/labels/area/iscrowd).
5. Prediction visualization panel (GT vs predictions with scores).
6. Training loss curve and runtime curve.
7. One failure-case prediction example with short diagnosis.
