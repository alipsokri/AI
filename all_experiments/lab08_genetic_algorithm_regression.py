"""
Lab 08 — Genetic Algorithms: Regression and Reproducibility

"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# 0. Locate project folders
# Works when VS Code starts in the project root or all_experiments.
# ---------------------------------------------------------------------
_candidates = [
    Path.cwd(),
    Path.cwd().parent,
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
]

ROOT = next(
    (p.resolve() for p in _candidates if (p / "all_datasets").is_dir()),
    Path(__file__).resolve().parent.parent,
)

DATA = ROOT / "all_datasets"
RESULTS = ROOT / "results" / "lab08"
RESULTS.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA / "lab08-insurance.csv"


def check_dataset():
    """Check that the expected dataset and columns are available."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}\n\n"
            "Place lab08-insurance.csv inside the all_datasets folder."
        )

    df = pd.read_csv(DATA_FILE)
    required = {"age", "bmi", "children", "smoker", "charges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("The required dataset columns contain missing values.")
    return df


# ---------------------------------------------------------------------
# Question 1 — Verify the evolutionary loop on f(x) = (x - 3)^2
# ---------------------------------------------------------------------
def ga_scalar(seed=42, pop_size=40, generations=80, n_elite=2):
    rng = np.random.default_rng(seed)
    population = rng.uniform(-10, 10, pop_size)

    for _ in range(generations):
        fitness = (population - 3) ** 2
        elite = population[np.argsort(fitness)[:n_elite]]

        # Tournament selection: best of three randomly sampled candidates.
        contestants = rng.integers(0, pop_size, size=(pop_size, 3))
        parents = population[
            contestants[np.arange(pop_size), fitness[contestants].argmin(axis=1)]
        ]

        # Blend crossover, followed by Gaussian mutation.
        a, b = parents[0::2], parents[1::2]
        alpha = rng.random(len(a))
        children = np.r_[
            alpha * a + (1 - alpha) * b,
            alpha * b + (1 - alpha) * a,
        ][: pop_size - n_elite]
        children += rng.normal(0, 0.15, len(children))
        population = np.r_[elite, children]

    best = population[np.argmin((population - 3) ** 2)]
    return best, (best - 3) ** 2


# ---------------------------------------------------------------------
# Question 2 — Prepare the medical-cost regression task
# ---------------------------------------------------------------------
def prepare_data(insurance):
    X = insurance[["age", "bmi", "children"]].assign(
        smoker=(insurance["smoker"].astype(str).str.lower() == "yes").astype(float)
    )
    y = insurance["charges"].to_numpy(dtype=float)

    train, test = train_test_split(
        insurance.index, test_size=0.20, random_state=42
    )

    split_labels = pd.Series(
        np.where(insurance.index.isin(test), "test", "train"),
        name="split",
    )
    split_labels.rename_axis("row_id").to_csv(RESULTS / "split.csv")

    # Fit scaler on training data only to avoid test-data leakage.
    scaler = StandardScaler().fit(X.loc[train])
    X_train = scaler.transform(X.loc[train])
    X_test = scaler.transform(X.loc[test])
    y_train = y[train]
    y_test = y[test]

    print(
        f"Rows: {len(insurance)} | Train: {len(train)} | Test: {len(test)}"
    )
    print("Predictors:", list(X.columns))
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# Question 3 — Genetic algorithm for regression coefficients
# ---------------------------------------------------------------------
def run_ga(
    seed,
    X_train,
    X_test,
    y_train,
    y_test,
    pop_size=100,
    generations=160,
    mutation_rate=0.15,
    mutation_scale=0.25,
    n_elite=4,
):
    rng = np.random.default_rng(seed)
    n_genes = X_train.shape[1] + 1

    population = rng.normal(0, 5000, size=(pop_size, n_genes))
    population[:, 0] += y_train.mean()
    gene_scale = np.r_[2000, np.repeat(1000, n_genes - 1)]

    def training_mse(individuals):
        predictions = individuals[:, [0]] + individuals[:, 1:] @ X_train.T
        return ((predictions - y_train) ** 2).mean(axis=1)

    best = None
    best_mse = np.inf
    history = []
    started = time.perf_counter()

    for _ in range(generations):
        fitness = training_mse(population)

        if fitness.min() < best_mse:
            best_mse = fitness.min()
            best = population[fitness.argmin()].copy()

        history.append(best_mse)

        # Elitism.
        elite = population[np.argsort(fitness)[:n_elite]]

        # Tournament selection (tournament size = 3).
        contestants = rng.integers(0, pop_size, size=(pop_size, 3))
        parents = population[
            contestants[np.arange(pop_size), fitness[contestants].argmin(axis=1)]
        ]

        # Blend crossover.
        a, b = parents[0::2], parents[1::2]
        alpha = rng.uniform(-0.1, 1.1, size=(len(a), 1))
        children = np.vstack([
            alpha * a + (1 - alpha) * b,
            alpha * b + (1 - alpha) * a,
        ])[: pop_size - n_elite]

        # Gaussian mutation on a random 15% of genes.
        mutate = rng.random(children.shape) < mutation_rate
        children += (
            mutate
            * rng.normal(0, mutation_scale, size=children.shape)
            * gene_scale
        )

        population = np.vstack([elite, children])

    test_predictions = best[0] + X_test @ best[1:]
    test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5

    return {
        "seed": seed,
        "train_rmse": best_mse ** 0.5,
        "test_rmse": test_rmse,
        "runtime_s": time.perf_counter() - started,
        "coefficients": best,
        "history": np.sqrt(history),
    }


def run_table(runs):
    return pd.DataFrame(runs)[
        ["seed", "train_rmse", "test_rmse", "runtime_s"]
    ]


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main():
    print("=" * 68)
    print("LAB 08 — GENETIC ALGORITHMS: REGRESSION AND REPRODUCIBILITY")
    print("=" * 68)
    print("Project root:", ROOT)
    print("Dataset:", DATA_FILE)
    print("Results folder:", RESULTS)

    insurance = check_dataset()

    print("\nQUESTION 1 — Scalar sanity check")
    x_best, f_best = ga_scalar()
    print(f"Best x = {x_best:.6f} | f(x) = {f_best:.8f}")

    print("\nQUESTION 2 — Prepare regression data")
    X_train, X_test, y_train, y_test = prepare_data(insurance)

    print("\nQUESTION 3 — One GA trial")
    trial = run_ga(1, X_train, X_test, y_train, y_test)
    print(
        f"Seed 1: train RMSE {trial['train_rmse']:.2f} | "
        f"test RMSE {trial['test_rmse']:.2f} | "
        f"runtime {trial['runtime_s']:.3f} s"
    )
    print(
        "Best chromosome [intercept, age, bmi, children, smoker]:",
        trial["coefficients"].round(1),
    )

    print("\nQUESTION 4 — Ten independent seeds")
    runs = [
        run_ga(seed, X_train, X_test, y_train, y_test)
        for seed in range(1, 11)
    ]
    results = run_table(runs)
    results.to_csv(RESULTS / "ga_10_runs.csv", index=False)
    print(results.round(3).to_string(index=False))
    print(
        f"Median test RMSE: {results.test_rmse.median():.2f} | "
        f"SD across seeds: {results.test_rmse.std():.4f}"
    )

    print("\nQUESTION 5 — Compare with ordinary least squares")
    ols = LinearRegression().fit(X_train, y_train)
    ols_test_rmse = mean_squared_error(y_test, ols.predict(X_test)) ** 0.5
    ols_train_rmse = mean_squared_error(y_train, ols.predict(X_train)) ** 0.5

    print("OLS test RMSE:", round(ols_test_rmse, 2))
    print("OLS coefficients:", ols.coef_.round(1))
    print(
        "Median GA coefficients:",
        np.median([r["coefficients"][1:] for r in runs], axis=0).round(1),
    )

    # Plot convergence for the first five seeds and the OLS training RMSE.
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in runs[:5]:
        ax.plot(
            result["history"],
            alpha=0.75,
            label=f"Seed {result['seed']}",
        )
    ax.axhline(
        ols_train_rmse,
        linestyle="--",
        color="black",
        label="OLS train RMSE",
    )
    ax.set(
        xlabel="Generation",
        ylabel="Best-so-far training RMSE",
        title="Genetic Algorithm convergence",
    )
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "convergence.png", dpi=160)
    plt.close(fig)
    print("Saved convergence plot.")

    print("\nQUESTION 6 — Population-size sensitivity (40)")
    small_runs = [
        run_ga(seed, X_train, X_test, y_train, y_test, pop_size=40, n_elite=2)
        for seed in range(1, 11)
    ]
    small = run_table(small_runs)
    small.to_csv(RESULTS / "ga_sensitivity_pop40.csv", index=False)

    summary = pd.DataFrame([
        {
            "method": "GA pop=100",
            "median_test_rmse": results.test_rmse.median(),
            "sd_test_rmse": results.test_rmse.std(),
            "median_runtime_s": results.runtime_s.median(),
        },
        {
            "method": "GA pop=40",
            "median_test_rmse": small.test_rmse.median(),
            "sd_test_rmse": small.test_rmse.std(),
            "median_runtime_s": small.runtime_s.median(),
        },
        {
            "method": "OLS",
            "median_test_rmse": ols_test_rmse,
            "sd_test_rmse": 0.0,
            "median_runtime_s": np.nan,
        },
    ])
    summary.to_csv(RESULTS / "summary.csv", index=False)

    print("\nSUMMARY")
    print(summary.round(4).to_string(index=False))

    print("\nInterpretation:")
    print("- Compare the median and standard deviation across all ten GA seeds.")
    print("- Matching OLS means the GA found the same optimum; it did not beat OLS.")
    print("- OLS is much faster for this unconstrained linear least-squares problem.")
    print("- A smaller population may run faster but can increase run-to-run variation.")

    print("\nAll outputs saved in:", RESULTS)
    print("Files:", sorted(p.name for p in RESULTS.iterdir()))


if __name__ == "__main__":
    main()
