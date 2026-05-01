# Final Report Result-Discussion Template

Use these figures/tables from `outputs_notebook/`:
- `loss_curves.png`
- `epoch_runtime.png`
- `val_pr_curve.png`
- `val_best_worst_examples.png`

## 1. Quantitative Summary
Report threshold settings:
- score threshold = `SCORE_THRESHOLD`
- IoU threshold = 0.5

Include:
- TP, FP, FN totals
- Precision, Recall, F1
- AP behavior from PR curve (`val_pr_curve.png`)

Suggested paragraph:
"On the validation split, the detector achieved [Precision], [Recall], and [F1] at score threshold [x] and IoU 0.5. The PR curve shows [shape/behavior], indicating [tradeoff observation]."

## 2. Success Cases
Use right-side "Best" examples from `val_best_worst_examples.png`.
Comment on:
- sparse scenes vs dense scenes
- box localization quality
- confidence calibration

Suggested paragraph:
"In high-quality cases, the detector captures most wheat heads with limited false positives, especially in [scene type]. Predicted confidence is relatively stable when object scale and texture are consistent."

## 3. Failure Cases
Use left-side "Worst" examples from `val_best_worst_examples.png`.
Comment on:
- missed detections in dense clusters (FN-heavy)
- over-detection in textured background (FP-heavy)
- overlap/occlusion effects

Suggested paragraph:
"Failure cases are dominated by [FN/FP]. In dense areas, heavy overlap causes missed detections; in textured regions, the model produces extra boxes. This indicates that crowding and local texture ambiguity remain key limitations."

## 4. Concrete Next Improvements (for final section)
1. Tune score threshold and NMS threshold using PR curve behavior.
2. Add denser-scene augmentation and/or crop-based training.
3. Compare one stronger backbone or detector variant against this baseline.
4. Keep the same split/seed for fair comparisons.
