"""Lab 3 of 8 — Search Engineering: Package, Tests, CLI, and Benchmark
Run this directly in VS Code: click "Run Python File" (top-right play button)
or type `python lab03_search_engineering.py` in the integrated terminal.

Expected location: all_experiments/lab03_search_engineering.py
(next to the utils/ folder, so `import utils` works)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from utils import SearchResult, bfs, dfs, ucs, gbfs, astar, haversine_km, load_openflights

pd.set_option("display.width", 200)                # show every column in the terminal
pd.set_option("display.max_columns", None)

HERE = Path(__file__).resolve().parent             # the all_experiments folder
ROOT = HERE.parent                                 # the AI course folder
DATA = ROOT / "all_datasets"
RESULTS = ROOT / "results" / "lab03"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Question 1 — Standardise the interface
# ---------------------------------------------------------------------------
print("\n=== Question 1: Same interface for all five algorithms ===")
print("SearchResult fields:", list(SearchResult.__dataclass_fields__))

toy = {"S": [("A", 2.0), ("B", 5.0)], "A": [("G", 5.0)], "B": [("G", 1.0)], "G": []}
h = {"S": 5.0, "A": 4.0, "B": 1.0, "G": 0.0}.get      # a hand-made consistent heuristic

results = {
    "BFS": bfs(toy, "S", "G"),
    "DFS": dfs(toy, "S", "G"),
    "UCS": ucs(toy, "S", "G"),
    "GBFS": gbfs(toy, "S", "G", h),
    "A*": astar(toy, "S", "G", h),
}
table = pd.DataFrame({name: r.as_dict() for name, r in results.items()}).T.drop(columns="runtime_ms")
table["path"] = table.path.map(" -> ".join)
print(table)


# ---------------------------------------------------------------------------
# Question 2 — Separate responsibilities
# ---------------------------------------------------------------------------
print("\n=== Question 2: What each module in utils/ is for ===")
package = Path(utils.__file__).parent
for module in sorted(package.glob("*.py")):
    purpose = module.read_text(encoding="utf-8").splitlines()[0].strip('"')
    print(f"{module.name:<16} {purpose}")
print("\nExported names:", ", ".join(utils.__all__))


# ---------------------------------------------------------------------------
# Question 3 — Run the automated tests
# ---------------------------------------------------------------------------
print("\n=== Question 3: Run the automated tests (pytest) ===")
tests = subprocess.run(
    [sys.executable, "-m", "pytest", "utils", "-q", "--color=no"],
    capture_output=True, text=True, cwd=HERE,      # cwd=HERE: run from all_experiments, wherever VS Code started
)
print(tests.stdout)
assert tests.returncode == 0, tests.stderr         # a failing test run stops the lab


# ---------------------------------------------------------------------------
# Question 4 — Exercise the command-line interface
# ---------------------------------------------------------------------------
print("\n=== Question 4: Run the command-line tool ===")
cli = subprocess.run(
    [sys.executable, "-m", "utils", "--algorithm", "astar", "--start", "KUL", "--goal", "KEF"],
    capture_output=True, text=True, cwd=HERE,
)
print(cli.stdout)
assert cli.returncode == 0, cli.stderr


# ---------------------------------------------------------------------------
# Question 5 — Benchmark ten route pairs
# ---------------------------------------------------------------------------
print("\n=== Question 5: Benchmark BFS, UCS, GBFS and A* on ten queries ===")
_, _, coords, graph, _ = load_openflights(DATA / "lab01-03_airports.csv", DATA / "lab01-03_routes.csv")
queries = pd.read_csv(HERE / "utils" / "search_queries.csv")      # all ten queries this time


def heuristic_to(goal):
    return lambda a: haversine_km(
        coords[a]["latitude"], coords[a]["longitude"],
        coords[goal]["latitude"], coords[goal]["longitude"],
    )


rows = []
for query in queries.itertuples():
    h = heuristic_to(query.goal)
    results = {
        "BFS": bfs(graph, query.start, query.goal),
        "UCS": ucs(graph, query.start, query.goal),
        "GBFS": gbfs(graph, query.start, query.goal, h),
        "A*": astar(graph, query.start, query.goal, h),
    }
    for name, result in results.items():
        rows.append({
            "query_id": query.query_id, "start": query.start, "goal": query.goal,
            "algorithm": name, **result.as_dict(), "path": " -> ".join(result.path),
        })

benchmark = pd.DataFrame(rows).rename(columns={"path_cost_km": "cost_km"})
benchmark.to_csv(RESULTS / "benchmark.csv", index=False)

summary = benchmark.groupby("algorithm").agg(
    success_rate=("found", "mean"),
    mean_cost_km=("cost_km", "mean"),
    mean_hops=("hops", "mean"),
    mean_expanded=("expanded", "mean"),
    median_runtime_ms=("runtime_ms", "median"),
).round(2)
summary.to_csv(RESULTS / "benchmark_summary.csv")
print(summary)

costs = benchmark.pivot(index="query_id", columns="algorithm", values="cost_km")
print("A* cost equals UCS on all 10 queries:", np.allclose(costs["A*"], costs["UCS"]))
print("GBFS is costlier than UCS on", int((costs["GBFS"] > costs["UCS"] + 1e-6).sum()), "of 10 queries")


# ---------------------------------------------------------------------------
# Question 6 — Save the evidence
# ---------------------------------------------------------------------------
print("\n=== Question 6: Save the evidence ===")
ax = summary.mean_expanded.sort_values().plot.bar(figsize=(7, 4), logy=True)
ax.set(ylabel="Mean expanded states (log scale)", title="Ten-query search-effort benchmark")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS / "benchmark_expansions.png", dpi=160)

print("Saved:", sorted(p.name for p in RESULTS.iterdir()))
print("Terminal command (from all_experiments): python -m pytest utils -q")
print("\nLab 3 complete.")
