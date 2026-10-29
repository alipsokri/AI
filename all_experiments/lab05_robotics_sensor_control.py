"""
Lab 05 — Robotics: Sensor-to-Action Control
Converted from lab05_robotics_sensor_control_worked.ipynb for use in VS Code.

Dataset expected:
    all_datasets/lab05-sensor_readings_24.csv

The CSV has no header: 24 ultrasonic distances (metres), followed by an action label.

Install dependencies:
    python -m pip install numpy pandas matplotlib scikit-learn

Run from the project folder:
    python lab05_robotics_sensor_control.py

Or specify paths:
    python lab05_robotics_sensor_control.py --data path/to/lab05-sensor_readings_24.csv
    python lab05_robotics_sensor_control.py --results path/to/results/lab05
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    script_dir = Path(__file__).resolve().parent
    default_candidates = [
        Path.cwd() / "all_datasets" / "lab05-sensor_readings_24.csv",
        script_dir / "all_datasets" / "lab05-sensor_readings_24.csv",
        script_dir.parent / "all_datasets" / "lab05-sensor_readings_24.csv",
    ]
    default_data = next((p for p in default_candidates if p.exists()), default_candidates[0])

    parser = argparse.ArgumentParser(
        description="Train and evaluate the Lab 5 sensor-to-action robot controller."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help="Path to lab05-sensor_readings_24.csv (default: auto-detected all_datasets folder).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path.cwd() / "results" / "lab05",
        help="Directory where CSV reports, rules, and plots will be saved.",
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    results_dir = args.results.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        print(f"ERROR: Dataset not found:\n  {data_path}", file=sys.stderr)
        print(
            "\nPlace 'lab05-sensor_readings_24.csv' inside an 'all_datasets' folder, "
            "or run with:\n"
            '  python lab05_robotics_sensor_control.py --data "FULL/PATH/TO/lab05-sensor_readings_24.csv"',
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print_section("Lab 6 — Robotics: Sensor-to-Action Control")
    print(f"Dataset: {data_path}")
    print(f"Results directory: {results_dir}")

    # ------------------------------------------------------------------
    # Question 1— Inspect the data and fix the split
    # ------------------------------------------------------------------
    sensors = [f"US{i:02d}" for i in range(24)]
    df = pd.read_csv(data_path, header=None, names=sensors + ["action"])

    if df.shape[1] != 25:
        raise ValueError(f"Expected 25 columns (24 sensors + action), found {df.shape[1]}.")
    if df.empty:
        raise ValueError("The dataset is empty.")
    if df.isna().any().any():
        raise ValueError(
            "The dataset contains missing values. Check the CSV before training the models."
        )

    X = df[sensors]
    y = df["action"]

    print_section("Question 1 — Inspect the data and fix the split")
    print("Dataset shape:", df.shape)
    class_balance = (
        y.value_counts()
        .rename("count")
        .to_frame()
        .assign(share=lambda table: (table["count"] / len(df)).round(3))
    )
    print("\nAction class balance:")
    print(class_balance.to_string())

    print("\nSensor summary statistics (first six sensors):")
    print(X.describe().T[["mean", "std", "min", "max"]].round(2).head(6).to_string())

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.30, random_state=42, stratify=y
    )
    split = pd.Series("train", index=df.index, name="split")
    split.loc[test_idx] = "test"
    split.rename_axis("row_id").to_csv(results_dir / "split.csv")
    print(f"\nSplit sizes: train={len(train_idx)}, test={len(test_idx)}")
    print(f"Saved: {results_dir / 'split.csv'}")

    # ------------------------------------------------------------------
    # Question 2 — Feature selection
    # ------------------------------------------------------------------
    print_section("Question 2 — Select four sensors using training data only")
    mi = mutual_info_classif(X.loc[train_idx], y.loc[train_idx], random_state=42)
    feature_table = pd.DataFrame(
        {"sensor": sensors, "mutual_information": mi}
    ).sort_values("mutual_information", ascending=False)
    feature_table.to_csv(results_dir / "feature_scores.csv", index=False)
    selected = feature_table["sensor"].head(4).tolist()

    print("Top eight sensors by mutual information:")
    print(feature_table.head(8).round(3).to_string(index=False))
    print("\nSelected sensors:", selected)
    print(f"Saved: {results_dir / 'feature_scores.csv'}")

    # ------------------------------------------------------------------
    # Question 3 — Interpretable decision-tree controller
    # ------------------------------------------------------------------
    print_section("Question 3 — Fit an interpretable rule controller")
    subtrain_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.20,
        random_state=42,
        stratify=y.loc[train_idx],
    )

    depth_results = []
    for depth in [2, 3, 4, 5, 6]:
        candidate = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        )
        candidate.fit(X.loc[subtrain_idx, selected], y.loc[subtrain_idx])
        validation_prediction = candidate.predict(X.loc[val_idx, selected])
        score = f1_score(
            y.loc[val_idx], validation_prediction, average="macro"
        )
        depth_results.append(
            {"max_depth": depth, "validation_macro_f1": score}
        )

    depth_table = pd.DataFrame(depth_results)
    depth_table.to_csv(results_dir / "tree_depth_validation.csv", index=False)
    print("Validation macro F1 by tree depth:")
    print(depth_table.round(3).to_string(index=False))

    best_depth = int(
        depth_table.loc[depth_table["validation_macro_f1"].idxmax(), "max_depth"]
    )
    rule_model = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    rule_model.fit(X.loc[train_idx, selected], y.loc[train_idx])

    rules = export_text(rule_model, feature_names=selected)
    (results_dir / "rule_controller.txt").write_text(rules, encoding="utf-8")
    print(f"\nSelected tree depth: {best_depth}")
    print(f"Number of leaves: {rule_model.get_n_leaves()}")
    print("\nExported decision rules:\n")
    print(rules)
    print(f"Saved: {results_dir / 'rule_controller.txt'}")

    # ------------------------------------------------------------------
    # Question 4 — Random Forest reference model
    # ------------------------------------------------------------------
    print_section("Question 4 — Fit a reference model")
    reference = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    reference.fit(X.loc[train_idx], y.loc[train_idx])
    print(
        "Random forest trained on",
        reference.n_features_in_,
        "sensors with",
        reference.n_estimators,
        "trees.",
    )

    # ------------------------------------------------------------------
    # Question 5 — Held-out evaluation
    # ------------------------------------------------------------------
    print_section("Question 5 — Evaluate on the held-out data")
    y_test = y.loc[test_idx]
    predictions = {
        "Interpretable rule tree": rule_model.predict(X.loc[test_idx, selected]),
        "Random-forest reference": reference.predict(X.loc[test_idx]),
    }

    metric_rows = []
    for name, prediction in predictions.items():
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            y_test, prediction, average="macro", zero_division=0
        )
        metric_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, prediction),
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": macro_f1,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    print("Test metrics:")
    print(metrics.round(3).to_string(index=False))
    print(f"Saved: {results_dir / 'metrics.csv'}")

    labels = sorted(y.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, prediction) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, prediction, labels=labels),
            display_labels=labels,
        ).plot(ax=ax, colorbar=False, xticks_rotation=35, values_format="d")
        ax.set_title(name)
    fig.tight_layout()
    confusion_path = results_dir / "confusion_matrices.png"
    fig.savefig(confusion_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {confusion_path}")

    precision, recall, class_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions["Interpretable rule tree"],
        labels=labels,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": class_f1},
        index=labels,
    )
    per_class.index.name = "action"
    per_class.to_csv(results_dir / "rule_per_class.csv")
    print("\nPer-class scores for the rule tree:")
    print(per_class.round(3).to_string())
    hardest_action = per_class["recall"].idxmin()
    print("\nHardest action for the rule controller (lowest recall):", hardest_action)
    print(f"Saved: {results_dir / 'rule_per_class.csv'}")

    # ------------------------------------------------------------------
    # Question 6 — Robustness to sensor noise
    # ------------------------------------------------------------------
    print_section("Question 6 — Test robustness to sensor noise")
    rng = np.random.default_rng(42)
    train_sd = X.loc[train_idx, selected].std().to_numpy()
    clean = X.loc[test_idx, selected].to_numpy(dtype=float)

    noise_rows = []
    for fraction in [0.0, 0.02, 0.05, 0.10]:
        noisy = clean + rng.normal(
            0.0, fraction * train_sd, size=clean.shape
        )
        noisy_frame = pd.DataFrame(noisy, columns=selected)
        prediction = rule_model.predict(noisy_frame)
        score = f1_score(y_test, prediction, average="macro")
        noise_rows.append(
            {
                "noise_fraction_of_train_sd": fraction,
                "noise_percent": fraction * 100,
                "macro_f1": score,
            }
        )

    noise = pd.DataFrame(noise_rows)
    noise.to_csv(results_dir / "noise_robustness.csv", index=False)
    print("Noise robustness results:")
    print(noise.round(3).to_string(index=False))
    print(f"Saved: {results_dir / 'noise_robustness.csv'}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        noise["noise_percent"],
        noise["macro_f1"],
        marker="o",
    )
    ax.set(
        xlabel="Gaussian noise (% of training-set standard deviation)",
        ylabel="Macro F1",
        title="Rule-controller robustness to sensor noise",
        xticks=[0, 2, 5, 10],
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    noise_plot_path = results_dir / "noise_robustness.png"
    fig.savefig(noise_plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {noise_plot_path}")

    print_section("Completed")
    print("All six lab questions have been run.")
    print("Results saved in:", results_dir)
    print("Files created:")
    for output_path in sorted(results_dir.iterdir()):
        if output_path.is_file():
            print(" -", output_path.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
