"""Lab 4 of 8 — Fuzzy Logic and ANFIS with Air-Quality Data
Run this directly in VS Code: click "Run Python File" (top-right play button)
or type `python lab04_fuzzy_anfis.py` in the integrated terminal.

Expected location: all_experiments/lab04_fuzzy_anfis.py
Needs:             all_datasets/lab04-AirQuality.csv
Saves evidence to: results/lab04/
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)                # show every column in the terminal
pd.set_option("display.max_columns", None)

ROOT = Path(__file__).resolve().parent.parent      # the AI course folder
DATA = ROOT / "all_datasets"
RESULTS = ROOT / "results" / "lab04"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# %% Question 1 — Clean the data
# ---------------------------------------------------------------------------
print("\n=== Question 1: Clean the data ===")
# The file uses ';' between columns and ',' as the decimal point.
df = pd.read_csv(DATA / "lab04-AirQuality.csv", sep=";", decimal=",")
df = df.dropna(axis="columns", how="all").dropna(how="all")   # empty unnamed columns, empty tail rows
df = df.replace(-200, np.nan)                                 # -200 means "no reading"

print("Rows:", len(df))
usable = df[["T", "RH", "CO(GT)", "PT08.S1(CO)", "PT08.S2(NMHC)"]].notna().sum()
print(usable.rename("usable rows").to_frame())

ax = df.plot.scatter(x="T", y="RH", s=7, alpha=0.25, figsize=(6, 4))
ax.set(xlabel="Temperature (deg C)", ylabel="Relative humidity (%)", title="Operating conditions")
plt.tight_layout()
plt.savefig(RESULTS / "temperature_humidity.png", dpi=160)
plt.close()


# ---------------------------------------------------------------------------
# %% Question 2 — Build the fuzzy controller (Mamdani)
# ---------------------------------------------------------------------------
print("\n=== Question 2: Build the fuzzy controller ===")


def trimf(x, a, b, c):
    """Triangle with feet a and c and peak b. a == b or b == c makes a shoulder."""
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else np.clip((x - a) / (b - a), 0, 1)
    right = np.ones_like(x) if b == c else np.clip((c - x) / (c - b), 0, 1)
    return np.minimum(left, right)


T_range, RH_range, V_range = np.linspace(0, 45, 451), np.linspace(10, 90, 401), np.linspace(0, 100, 501)

# Try this: change the 26 in "hot" to 20, then rerun Question 3 and compare.
T_sets = {"cold": trimf(T_range, 0, 0, 18), "mild": trimf(T_range, 12, 22, 32), "hot": trimf(T_range, 26, 45, 45)}
RH_sets = {"dry": trimf(RH_range, 10, 10, 40), "comfortable": trimf(RH_range, 30, 50, 70), "humid": trimf(RH_range, 60, 90, 90)}
V_sets = {"low": trimf(V_range, 0, 0, 45), "medium": trimf(V_range, 25, 50, 75), "high": trimf(V_range, 55, 100, 100)}


def degree(universe, membership, value):
    """Membership degree of one crisp value."""
    return float(np.interp(value, universe, membership))


def ventilation(temperature, humidity):
    # 1. Fuzzify: how much does each reading belong to each set?
    t = {name: degree(T_range, mf, temperature) for name, mf in T_sets.items()}
    h = {name: degree(RH_range, mf, humidity) for name, mf in RH_sets.items()}
    # 2. Fire the rules: AND is min; rules with the same conclusion combine with max.
    low = max(min(t["cold"], h["dry"]), min(t["cold"], h["comfortable"]))
    medium = max(min(t["mild"], h["dry"]), min(t["mild"], h["comfortable"]), min(t["cold"], h["humid"]))
    high = max(t["hot"], h["humid"])
    # 3. Clip each output set at its rule strength and take the union.
    output = np.maximum.reduce([
        np.minimum(low, V_sets["low"]),
        np.minimum(medium, V_sets["medium"]),
        np.minimum(high, V_sets["high"]),
    ])
    # 4. Defuzzify: the centroid (balance point) of the combined shape.
    return float(np.sum(V_range * output) / np.sum(output))


fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (title, universe, sets) in zip(axes, [("Temperature", T_range, T_sets),
                                              ("Humidity", RH_range, RH_sets),
                                              ("Ventilation", V_range, V_sets)]):
    for name, membership in sets.items():
        ax.plot(universe, membership, label=name)
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "fuzzy_memberships.png", dpi=160)
plt.close()

print("Worked example from the lecture: 15 deg C, 45 % humidity ->",
      round(ventilation(15, 45), 1), "% ventilation")
print("Try it yourself: 28 deg C, 60 % humidity ->", round(ventilation(28, 60), 1), "% ventilation")


# ---------------------------------------------------------------------------
# %% Question 3 — Validate the rule base
# ---------------------------------------------------------------------------
print("\n=== Question 3: Validate the rule base ===")
fixed = pd.DataFrame({"T_C": [10, 20, 35, 25, 40], "RH_pct": [30, 50, 50, 80, 80],
                      "expected": ["low", "medium", "high", "high", "high"]})
fixed["ventilation_pct"] = [ventilation(t, h) for t, h in zip(fixed.T_C, fixed.RH_pct)]
fixed.to_csv(RESULTS / "fuzzy_fixed_points.csv", index=False)
print(fixed.round(1))

temperatures, humidities = np.linspace(0, 45, 46), np.linspace(10, 90, 41)
surface = np.array([[ventilation(t, h) for t in temperatures] for h in humidities])

fig, ax = plt.subplots(figsize=(7, 4.5))
contour = ax.contourf(temperatures, humidities, surface, levels=15)
fig.colorbar(contour, label="Ventilation demand (%)")
ax.set(xlabel="Temperature (deg C)", ylabel="Relative humidity (%)", title="Fuzzy control surface")
plt.tight_layout()
plt.savefig(RESULTS / "fuzzy_control_surface.png", dpi=160)
plt.close()


# ---------------------------------------------------------------------------
# %% Question 4 — Prepare the ANFIS task
# ---------------------------------------------------------------------------
print("\n=== Question 4: Prepare the ANFIS task ===")
task = df[["CO(GT)", "PT08.S1(CO)", "PT08.S2(NMHC)"]].dropna()      # complete rows only
X = task[["PT08.S1(CO)", "PT08.S2(NMHC)"]].to_numpy()
y = task["CO(GT)"].to_numpy()

train, rest = train_test_split(np.arange(len(task)), test_size=0.30, random_state=42)
val, test = train_test_split(rest, test_size=0.50, random_state=42)
split = pd.Series("train", index=task.index, name="split")
split.iloc[val], split.iloc[test] = "validation", "test"
split.rename_axis("source_row").to_csv(RESULTS / "split.csv")

scaler = StandardScaler().fit(X[train])           # training statistics only: no leakage
Xs = scaler.transform(X)
print({"usable_rows": len(task), "train": len(train), "validation": len(val), "test": len(test)})


# ---------------------------------------------------------------------------
# %% Question 5 — Fit two model classes (linear baseline and Sugeno ANFIS)
# ---------------------------------------------------------------------------
print("\n=== Question 5: Fit two model classes ===")


class SugenoANFIS:
    """First-order Sugeno ANFIS: Gaussian memberships, linear rule outputs."""

    # Initialization
    def __init__(self, n_mfs=2, ridge=1e-4, maxiter=30):
        self.n_mfs, self.ridge, self.maxiter = n_mfs, ridge, maxiter  # Saves 2 into model_A's memory, and 3 into model_B's memory!

    # Fuzzy Layer 1 to 3
    def _weights(self, X, theta):
        """Layers 1-3: memberships -> rule strengths -> normalised strengths."""
        n_features = X.shape[1]
        centers = theta[: n_features * self.n_mfs].reshape(n_features, self.n_mfs) # Gaussian bell curves. he code reshapes this flat array into a grid matrix (n_features * n_mfs)
        widths = np.exp(theta[n_features * self.n_mfs:]).reshape(n_features, self.n_mfs) 
        # The second half holds the spread (standard deviation) of the curves. The code uses np.exp() to guarantee that these widths stay positive numbers during training, since a negative width would break the math formulas.
        membership = [np.exp(-0.5 * ((X[:, [j]] - centers[j]) / widths[j]) ** 2) for j in range(n_features)]
        strength = np.column_stack([np.prod([membership[j][:, m] for j, m in enumerate(rule)], axis=0)
                                    for rule in product(range(self.n_mfs), repeat=n_features)])
        return strength / (strength.sum(axis=1, keepdims=True) + 1e-12)

    # Rule Consequent Layer 4
    def _design(self, X, theta):
        """Layer 4: every rule contributes weight x (1, x1, x2) to a linear output."""
        weights = self._weights(X, theta)
        inputs = np.column_stack([np.ones(len(X)), X])
        return np.hstack([weights[:, [r]] * inputs for r in range(weights.shape[1])])

    def _solve_outputs(self, X, y, theta):
        Z = self._design(X, theta)
        return np.linalg.solve(Z.T @ Z + self.ridge * np.eye(Z.shape[1]), Z.T @ y) #Regularized Least Squares (Ridge Regression)
        # @: This is Python's symbol for matrix multiplication.
        # Z.T: This is the transpose of matrix Z (flipping rows into columns)

    # Training
    def fit(self, X, y):
        n_features = X.shape[1]
        centers = np.array([np.quantile(X[:, j], np.linspace(0.25, 0.75, self.n_mfs)) for j in range(n_features)])
        widths = np.full((n_features, self.n_mfs), max(X.std(), 0.5))
        theta0 = np.r_[centers.ravel(), np.log(widths).ravel()]

        def training_mse(theta):
            return np.mean((y - self._design(X, theta) @ self._solve_outputs(X, y, theta)) ** 2)

        bounds = [(-3, 3)] * centers.size + [(np.log(0.15), np.log(3))] * widths.size
        self.theta_ = minimize(training_mse, theta0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": self.maxiter}).x
        self.outputs_ = self._solve_outputs(X, y, self.theta_)
        return self

    # Prediction Layer 5
    def predict(self, X):
        return self._design(X, self.theta_) @ self.outputs_      # layer 5: weighted sum


def scores(name, model):
    row = {"model": name}
    for part, rows in [("validation", val), ("test", test)]:
        prediction = model.predict(Xs[rows])
        row[f"{part}_rmse"] = mean_squared_error(y[rows], prediction) ** 0.5
        row[f"{part}_mae"] = mean_absolute_error(y[rows], prediction)
    return row


linear = LinearRegression().fit(Xs[train], y[train])
candidates = {n: SugenoANFIS(n_mfs=n).fit(Xs[train], y[train]) for n in [2, 3]}

metrics = pd.DataFrame([scores("Linear regression", linear)] +
                       [scores(f"ANFIS-{n}MF", m) for n, m in candidates.items()])
metrics.to_csv(RESULTS / "metrics.csv", index=False)
print(metrics.round(4))

# Select on VALIDATION RMSE only. The test columns must not influence this choice.
by_model = metrics.set_index("model")
best_n = min(candidates, key=lambda n: by_model.loc[f"ANFIS-{n}MF", "validation_rmse"])
anfis = candidates[best_n]
print(f"Selected by validation RMSE: ANFIS-{best_n}MF")


# ---------------------------------------------------------------------------
# %% Question 6 — Evaluate once on the test data
# ---------------------------------------------------------------------------
print("\n=== Question 6: Evaluate once on the test data ===")
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for ax, (name, model) in zip(axes, [("Linear baseline", linear), (f"ANFIS-{best_n}MF", anfis)]):
    prediction = model.predict(Xs[test])
    ax.scatter(y[test], prediction, s=10, alpha=0.35)
    ax.plot([0, y[test].max()], [0, y[test].max()], "--", linewidth=1)
    ax.set(title=name, xlabel="Actual CO(GT)", ylabel="Predicted CO(GT)")
plt.tight_layout()
plt.savefig(RESULTS / "predicted_vs_actual.png", dpi=160)
plt.close()

print(by_model.loc[["Linear regression", f"ANFIS-{best_n}MF"], ["test_rmse", "test_mae"]].round(4))
improvement = 1 - by_model.loc[f"ANFIS-{best_n}MF", "test_rmse"] / by_model.loc["Linear regression", "test_rmse"]
print(f"Test RMSE improvement over linear regression: {improvement:.1%}")
print("Saved:", sorted(p.name for p in RESULTS.iterdir()))
print("\nLab 4 complete.") 
