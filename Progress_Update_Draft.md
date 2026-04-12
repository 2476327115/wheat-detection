# Deep Learning for Wheat Detection in Aerial Images  
**CS444/ECE494 Four-Credit Project Progress Update**

## 1. Group Members
- Gezhi Zou

## 2. Project Title
**Deep Learning for Wheat Detection in Aerial Images**

## 3. Updated Problem Definition and Plan
The goal of this project remains the same as in the original proposal: build and evaluate a deep learning object detector that identifies wheat heads in aerial field images from the Global Wheat Detection dataset. Given an image, the model should output bounding boxes and confidence scores for detected wheat heads.

This remains a meaningful computer vision task because wheat heads are relatively small, can be densely packed, and often appear under varying lighting/background conditions. These characteristics make detection challenging and provide a good testbed for modern detection pipelines.

### Changes from Proposal
The core direction has not changed (Faster R-CNN + transfer learning), but the project plan has been refined in the following ways:
- The implementation now prioritizes building a reliable baseline first (default `torchvision` Faster R-CNN pipeline) before testing larger architectural changes.
- The evaluation plan now includes clearer milestone stages: (1) data loading sanity checks, (2) baseline training/inference, (3) error analysis, (4) ablation experiments.
- The training strategy now emphasizes reproducibility (fixed random seeds, fixed split, consistent logging) so later experiments are directly comparable.

These are refinements rather than major scope changes, and they were made to reduce execution risk and ensure measurable progress each week.

## 4. Illustrations of Data and Target Outputs
This section should include both dataset examples and model output examples.

### 4.1 Data Examples (Input)
The Global Wheat Detection dataset contains RGB aerial images with wheat-head bounding box annotations. Visual inspection shows:
- Wheat heads are often small relative to image size.
- Object density varies significantly by scene.
- Occlusion and overlap are common in dense regions.
- Contrast between wheat and background can vary due to lighting and field conditions.

**Figure 1 (to insert):** Raw training image example from dataset.  
**Figure 2 (to insert):** Same image with ground-truth bounding boxes.

Suggested caption text:
- *Figure 1. Example raw aerial field image from the Global Wheat Detection training set.*
- *Figure 2. Ground-truth wheat-head bounding boxes on the same image, showing dense-object detection difficulty.*

### 4.2 Target Output Examples
Target model outputs are predicted bounding boxes with confidence values. For each image, predictions should:
- Cover visible wheat heads.
- Avoid excessive false positives in background regions.
- Maintain stable localization quality in crowded areas.

**Figure 3 (to insert):** Baseline model predictions on validation image.  
**Figure 4 (to insert):** Failure-case example (missed dense region or false positives).

Suggested caption text:
- *Figure 3. Predicted wheat-head boxes from baseline detector on validation sample.*
- *Figure 4. Representative failure case highlighting dense-region misses and localization errors.*

## 5. Key Resources

### 5.1 Dataset
- Global Wheat Detection (Kaggle competition):  
  <https://www.kaggle.com/competitions/global-wheat-detection>

Data used:
- Training images
- Bounding-box annotations
- Validation split created from training set for local experiments

### 5.2 Code Bases and Libraries
- PyTorch: <https://pytorch.org>
- Torchvision detection models: <https://pytorch.org/vision/stable/models.html>
- Optional reference implementations in Kaggle notebooks (used for comparison and debugging ideas, not copy-paste integration)

### 5.3 Compute Platform
- Primary compute: local workstation with NVIDIA RTX 3080 Ti GPU
- Secondary option: Google Colab GPU runtime if additional long runs are needed

### 5.4 Build-vs-Reuse Clarification
- **From scratch:** data parsing, train/validation split logic, training loop structure, experiment tracking, evaluation scripts, and analysis pipeline.
- **Existing code reused:** base detector architecture from `torchvision` (Faster R-CNN implementation).
- **Pre-trained models:** ImageNet/COCO pre-trained backbone/detector weights for transfer learning initialization.
- **Training status:** model is fine-tuned on target wheat data rather than fully training from random initialization.

## 6. Summary of Work Done to Date
Progress so far focuses on establishing a stable end-to-end baseline pipeline.

### 6.1 Completed
- Verified dataset access and local file organization.
- Reviewed annotation format and confirmed conversion path to model input tensors.
- Defined a training/validation split strategy for controlled local evaluation.
- Set up baseline model choice (Faster R-CNN with ResNet backbone).
- Prepared the training environment with required deep learning dependencies.

### 6.2 Evidence of Running Code
Include concrete evidence in final submission (required):
- **Screenshot A (to insert):** Training log/terminal showing successful epoch execution.
- **Screenshot B (to insert):** Inference output visualization with predicted boxes on at least one validation image.
- **Screenshot C (optional):** GPU utilization or training runtime summary.

Suggested text you can keep and fill:
- Baseline run executed on [DATE] for [N] epochs.
- Final training loss after run: [VALUE].
- Validation metric used: [METRIC NAME], value: [VALUE].
- Number of qualitative validation images reviewed: [N].

### 6.3 Baseline Results (Fill with Real Numbers)
Use a concise table in your final PDF/doc:

| Experiment | Backbone | Epochs | Input Size | Validation Metric | Value |
|---|---|---:|---:|---|---:|
| Baseline-1 | ResNet-[X] | [N] | [H x W] | [mAP / competition metric] | [V] |
| Baseline-2 (if available) | ResNet-[X] | [N] | [H x W] | [mAP / competition metric] | [V] |

Current qualitative observations:
- Baseline can detect many isolated wheat heads.
- Performance drops in very dense clusters.
- Some false positives occur in high-texture background patches.

## 7. Plan for the Remainder of the Project

### 7.1 Minimum Goal
Deliver a complete and reproducible wheat detection pipeline with:
- One solid baseline model (Faster R-CNN)
- Quantitative evaluation on held-out validation split
- Qualitative visualization of successes/failures
- Brief ablation on one or two training settings (e.g., augmentation and learning rate)

### 7.2 Maximum Goal
Extend beyond baseline with stronger performance and clearer analysis:
- Compare two detector/backbone variants
- Tune confidence/NMS thresholds for better precision-recall tradeoff
- Explore targeted augmentation for dense-object scenes
- Produce stronger error taxonomy and mitigation discussion

### 7.3 Risks and Mitigation
- **Risk: Limited training time on full-resolution data.**  
  Mitigation: start with lower-resolution sanity runs; use mixed precision and shorter pilot runs for hyperparameter screening.
- **Risk: Overfitting due to limited effective variation.**  
  Mitigation: stronger augmentation, early stopping, and tighter validation monitoring.
- **Risk: Weak performance in crowded regions.**  
  Mitigation: threshold tuning, improved sampling/augmentation, and targeted analysis of dense-scene errors.
- **Risk: Metric instability from split randomness.**  
  Mitigation: fixed seed and fixed split for core comparisons.

## 8. Member Roles and Collaboration Plan
Since this is currently a one-member project, responsibilities are centralized and tracked by milestone.

- **Gezhi Zou:**
  - Data preparation and preprocessing
  - Training/inference pipeline implementation
  - Experiment management and metric logging
  - Qualitative error analysis and visualization
  - Final report writing and presentation preparation

Collaboration workflow (individual version):
- Maintain a weekly milestone checklist.
- Record experiment settings and results after each run.
- Keep report text synchronized with actual experiment evidence (plots, logs, and tables).

If additional team members are added later, this section will be updated with role-specific ownership (data engineering, modeling, and evaluation/reporting splits).

## 9. Reference List
1. Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* <https://arxiv.org/abs/1506.01497>
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* <https://arxiv.org/abs/1512.03385>
3. Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* <https://pytorch.org>
4. Kaggle. (2026). *Global Wheat Detection.* <https://www.kaggle.com/competitions/global-wheat-detection>

---

## Final Submission Checklist (Quick)
- [ ] At least 4 full pages in final PDF/Doc format
- [ ] Insert 2+ dataset/annotation figures
- [ ] Insert 1+ model output figure and 1 failure-case figure
- [ ] Include training/inference evidence screenshot(s)
- [ ] Fill real baseline metric values in table
- [ ] Ensure all links and references are visible and consistent
