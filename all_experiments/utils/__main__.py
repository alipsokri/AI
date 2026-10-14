"""Command-line route search (Lab 3).

Run from the ``all_experiments`` folder::

    python -m utils --algorithm astar --start KUL --goal KEF
"""
import argparse
from pathlib import Path

from . import astar, bfs, dfs, gbfs, haversine_km, load_openflights, ucs

DATASETS = Path(__file__).resolve().parents[2] / "all_datasets"


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the OpenFlights route graph.")
    parser.add_argument("--algorithm", choices=["bfs", "dfs", "ucs", "gbfs", "astar"], default="astar")
    parser.add_argument("--start", required=True, help="origin IATA code, e.g. KUL")
    parser.add_argument("--goal", required=True, help="destination IATA code, e.g. KEF")
    args = parser.parse_args()
    start, goal = args.start.upper(), args.goal.upper()

    _, _, coords, graph, _ = load_openflights(DATASETS / "lab01-03_airports.csv", DATASETS / "lab01-03_routes.csv")
    if start not in coords or goal not in coords:
        parser.error(f"unknown airport code: {start if start not in coords else goal}")

    def h(airport):
        return haversine_km(coords[airport]["latitude"], coords[airport]["longitude"],
                            coords[goal]["latitude"], coords[goal]["longitude"])

    search = {"bfs": bfs, "dfs": dfs, "ucs": ucs}
    if args.algorithm in search:
        result = search[args.algorithm](graph, start, goal)
    else:
        result = (gbfs if args.algorithm == "gbfs" else astar)(graph, start, goal, h)

    print(f"{args.algorithm.upper()}: found={result.found}, hops={result.hops}, "
          f"cost_km={result.path_cost_km:.1f}, expanded={result.expanded}, runtime_ms={result.runtime_ms:.2f}")
    if result.found:
        print(" -> ".join(result.path))
    return 0 if result.found else 2


if __name__ == "__main__":
    raise SystemExit(main())
