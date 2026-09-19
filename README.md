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
| `alns_cars_noneuclidean.py` | Adaptive Large Neighborhood Search (ALNS) | Non-Euclidean |
| `alns_cars_euclidean.py` | Adaptive Large Neighborhood Search (ALNS) | Euclidean |

All four files are plain command-line scripts (no GUI, no plots) — each exposes a `run(instance_path, seed=42, verbose=False) -> dict` function (route, vehicles, deterministic cost, expected PTSP cost, elapsed time, best iteration) used both by their own `if __name__ == "__main__":` block and by `run_experiments.py` (see below) for batch benchmarking.

## Requirements

- Python 3.8+
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/) and [openpyxl](https://openpyxl.readthedocs.io/) (only needed for `run_experiments.py`)

Install the Python dependencies with:

```bash
pip install numpy pandas openpyxl
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

Each script defaults to one instance under `instances/` (hardcoded in the `if __name__ == "__main__":` block at the bottom of the file) — edit that line to point at a different `.car` file if you want.

## Batch benchmarking

`run_experiments.py` runs all 4 algorithms over a filtered set of instances, several times each, and writes an Excel (`.xlsx`) report:

```bash
python run_experiments.py --min-n 9 --max-n 17 --runs 10
python run_experiments.py --min-n 25 --max-n 52 --runs 10
python run_experiments.py --min-n 70 --max-n 300 --runs 5 --output resultados/grandes.xlsx
```

- `--min-n` / `--max-n` — only instances whose `DIMENSION` falls in this range (run in size-based blocks to keep individual invocations fast).
- `--runs` — repetitions per (algorithm, instance) pair; default 10.
- `--output` — output `.xlsx` path; defaults to an auto-named file under `resultados/`.

The output has two sheets:

- **Resultados** — one row per run: `algoritmo, instancia, N, K, corrida, semilla, tiempo_s, costo_determinista, costo_esperado, iter_mejor`.
- **Resumen** — one row per (algoritmo, instancia), aggregating the repetitions: `algoritmo, instancia, N, K, corridas, tiempo_prom_s, tiempo_min_s, tiempo_max_s, esperado_prom, esperado_min, esperado_max, esperado_std`.

Selection/optimization in all 4 algorithms is by **expected PTSP cost** (`costo_esperado`); `costo_determinista` is kept only as a reference value.
