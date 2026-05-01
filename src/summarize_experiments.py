import argparse
from pathlib import Path
import pandas as pd


def load_final_row(csv_path: Path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    row = df.iloc[-1].to_dict()
    row["csv_path"] = str(csv_path)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-out", type=str, default="C:/wheat-detection/outputs_experiments")
    parser.add_argument("--save", type=str, default="C:/wheat-detection/outputs_experiments/experiment_summary.csv")
    args = parser.parse_args()

    base = Path(args.base_out)
    rows = []
    for csv in base.glob("*/training_history.csv"):
        r = load_final_row(csv)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No training_history.csv files found.")
        return

    out = pd.DataFrame(rows)
    cols = [
        "experiment_tag",
        "backbone",
        "anchor_sizes",
        "use_scale_jitter",
        "use_random_crop",
        "loss_total",
        "val_loss_total",
        "train_map50",
        "val_map50",
        "val_avg_pred_boxes",
        "val_avg_gt_boxes",
        "epoch_time_sec",
        "csv_path",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values(by=[c for c in ["val_map50", "train_map50"] if c in out.columns], ascending=False)
    out.to_csv(args.save, index=False)
    print("Saved summary:", args.save)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
