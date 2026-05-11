param(
    [string]$DataDir = "C:\wheat-detection\data",
    [int]$Epochs = 10,
    [int]$BatchSize = 4,
    [int]$EvalEvery = 1,
    [int]$EarlyStopPatience = 3,
    [string]$BaseOut = "C:\wheat-detection\outputs_experiments"
)

New-Item -ItemType Directory -Force -Path $BaseOut | Out-Null

# 1) Baseline: default anchors, ResNet-50, basic augmentation (hflip/vflip)
python C:\wheat-detection\src\faster_rcnn_wheat.py `
  --data-dir $DataDir `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --eval-every $EvalEvery `
  --early-stop-patience $EarlyStopPatience `
  --backbone resnet50 `
  --experiment-tag baseline_r50_default `
  --output-dir "$BaseOut\baseline_r50_default"

# 2) Anchor customization for small objects
python C:\wheat-detection\src\faster_rcnn_wheat.py `
  --data-dir $DataDir `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --eval-every $EvalEvery `
  --early-stop-patience $EarlyStopPatience `
  --backbone resnet50 `
  --anchor-sizes "16,24,32,48,64" `
  --anchor-aspects "0.5,1.0,2.0" `
  --experiment-tag anchor_tuned_r50 `
  --output-dir "$BaseOut\anchor_tuned_r50"

# 3) Augmentation study: scale jitter + random crop
python C:\wheat-detection\src\faster_rcnn_wheat.py `
  --data-dir $DataDir `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --eval-every $EvalEvery `
  --early-stop-patience $EarlyStopPatience `
  --backbone resnet50 `
  --anchor-sizes "16,24,32,48,64" `
  --use-scale-jitter `
  --use-random-crop `
  --experiment-tag anchor_aug_r50 `
  --output-dir "$BaseOut\anchor_aug_r50"

# 4) Backbone comparison: ResNet-101 under tuned anchors + augmentation
python C:\wheat-detection\src\faster_rcnn_wheat.py `
  --data-dir $DataDir `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --eval-every $EvalEvery `
  --early-stop-patience $EarlyStopPatience `
  --backbone resnet101 `
  --anchor-sizes "16,24,32,48,64" `
  --use-scale-jitter `
  --use-random-crop `
  --experiment-tag anchor_aug_r101 `
  --output-dir "$BaseOut\anchor_aug_r101"

Write-Host "All proposal experiments finished. Outputs in: $BaseOut"
