# CaRS — Car Renting Salesman Problem Solvers

Metaheuristic solvers for the **Car Renting Salesman Problem (CaRS)**, based on Paulo Henrique Asconavieta da Silva's 2011 doctoral thesis *"O Problema do Caixeiro Alugador: Um Estudo Algorítmico"* (UFRN). CaRS extends the classic Traveling Salesman Problem: the tour can be split into contiguous segments, each driven by a different rental car, at the cost of a return fee for returning each car to the city where it was rented.

All four solvers work in two layers:

1. **Route search** (metaheuristic) proposes an order in which to visit the cities.
2. **Vehicle assignment** (exact dynamic program) finds the optimal way to split that fixed route across vehicles — under the **"exact"** variant (`exato` in the thesis): every available vehicle must be used exactly once, each in one contiguous block, and its return cost (rented-city → drop-off-city, per the thesis' `d^k_ij`) is charged when its block closes, including the last vehicle of the tour.

Once a final route is found, each solver also computes the **expected cost under the PTSP model** (probabilistic customer presence) with a closed-form O(n²) formula, and cross-validates it against an exhaustive brute-force enumeration over presence/absence scenarios.

## Files

| File | Search algorithm | Instance type |
|---|---|---|
| `aco_cars_ptsp_exact_noneuclidean.py` | Ant Colony System (ACS) + 2-opt/Or-opt | Non-Euclidean (explicit cost matrices) |
| `aco_cars_ptsp_exact_euclidean.py` | Ant Colony System (ACS) + 2-opt/Or-opt | Euclidean (2D coordinates) |
| `alns_cars_noneuclidean.py` | Adaptive Large Neighborhood Search (ALNS), with Tkinter GUI | Non-Euclidean |
| `alns_cars_euclidean.py` | Adaptive Large Neighborhood Search (ALNS), with Tkinter GUI | Euclidean |

The two ACO files run from the command line and print/plot their results. The two ALNS files launch a Tkinter GUI to load an instance, set search parameters, run/pause/stop the search, and export the resulting solution.

## Requirements

- Python 3.8+
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- `tkinter` (only needed for `alns_cars_noneuclidean.py` / `alns_cars_euclidean.py`) — bundled with the official Python installer on Windows and macOS; on Linux it's usually a separate package, e.g. `sudo apt install python3-tk`.

Install the Python dependencies with:

```bash
pip install numpy matplotlib
```

## Instance files

These scripts read `.car` instance files — a TSPLIB-inspired text format. A set of 13 non-Euclidean and 13 Euclidean CaRSLIB instances (real-map based, per the thesis) ships in this repo under:

```
instances/
  euclidean/      # for aco_cars_ptsp_exact_euclidean.py, alns_cars_euclidean.py
  noneuclidean/    # for aco_cars_ptsp_exact_noneuclidean.py, alns_cars_noneuclidean.py
```

Each instance file has:

- `DIMENSION`, `CARS_NUMBER` — number of cities and number of vehicles.
- `EDGE_WEIGHT_SECTION`, `RETURN_RATE_SECTION` — per-vehicle travel and return costs.
- `NODE_COORD_SECTION` — only in the Euclidean instances.
- `PROBABILITY_SECTION` (optional) — presence probability per city, for the PTSP expected-cost calculation; defaults to 1.0 (always present) for every city if omitted.

## Running

```bash
python aco_cars_ptsp_exact_noneuclidean.py
python aco_cars_ptsp_exact_euclidean.py
python alns_cars_noneuclidean.py
python alns_cars_euclidean.py
```

The two ACO scripts default to an instance under `instances/` (hardcoded in the `if __name__ == "__main__":` block at the bottom of the file) — edit that line to point at a different `.car` file if you want. The two ALNS scripts prompt you to select an instance file from the GUI, opening directly in the matching `instances/euclidean` or `instances/noneuclidean` folder.
