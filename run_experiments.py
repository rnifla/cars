"""
Batch benchmarking driver for the 4 CaRS solvers (2026-09).

Runs aco_cars_ptsp_exact_euclidean / aco_cars_ptsp_exact_noneuclidean /
alns_cars_euclidean / alns_cars_noneuclidean over a filtered set of .car
instances, several times each, and writes an Excel report with a raw
per-run sheet and a per-(algoritmo, instancia) summary sheet.

Every run is appended to a checkpoint CSV (same path as the .xlsx, with a
.csv extension) as soon as it finishes. Re-running the same command resumes
from that CSV, skipping runs already done, so a crash/reboot loses at most
the run in progress. The .xlsx is built from the CSV at the end.

Each algorithm's run(instance_path, seed, verbose) function (added
2026-09 to all 4 files, alongside removing their GUI/matplotlib code) is
the single entry point used here -- it does exactly one search with no
printing/plotting, and returns a result dict with cost, expected_cost,
elapsed_s, best_iteration, n, k.

Usage:
    python run_experiments.py --min-n 9 --max-n 17 --runs 10
    python run_experiments.py --min-n 25 --max-n 52 --runs 10
    python run_experiments.py --min-n 70 --max-n 300 --runs 5 --output resultados/grandes.xlsx

Instancias filtradas por rango de N (DIMENSION), para poder correr por
bloques de tamano (chicas / medianas / grandes) en llamadas separadas.
"""

import argparse
import glob
import os
import time

import pandas as pd

import aco_cars_ptsp_exact_euclidean as aco_e
import aco_cars_ptsp_exact_noneuclidean as aco_n
import alns_cars_euclidean as alns_e
import alns_cars_noneuclidean as alns_n

ALGORITHMS = {
    "ACO_euclidean": (aco_e.run, "instances/euclidean"),
    "ACO_noneuclidean": (aco_n.run, "instances/noneuclidean"),
    "ALNS_euclidean": (alns_e.run, "instances/euclidean"),
    "ALNS_noneuclidean": (alns_n.run, "instances/noneuclidean"),
}


def _peek_dimension(car_path: str) -> int:
    """Read just DIMENSION from a .car file, without building the full
    instance -- cheap enough to call once per file during discovery."""
    with open(car_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("DIMENSION"):
                return int(line.split(":")[1].strip())
    raise ValueError(f"DIMENSION not found in {car_path}")


def discover_instances(folder: str, min_n: int, max_n: int) -> list:
    """.car files in `folder` whose DIMENSION falls in [min_n, max_n],
    sorted by N then name. Skips non-.car files (e.g. stray .txt copies)."""
    paths = sorted(glob.glob(os.path.join(folder, "*.car")))
    result = []
    for p in paths:
        try:
            n = _peek_dimension(p)
        except (ValueError, OSError):
            continue
        if min_n <= n <= max_n:
            result.append((p, n))
    return result


def run_batch(min_n: int, max_n: int, n_runs: int, csv_path: str) -> pd.DataFrame:
    """Runs every algorithm over every matching instance n_runs times,
    appending each finished run to csv_path. Runs already present in the CSV
    are skipped (resume). Returns all raw per-run results as a DataFrame.
    Failures (e.g. an infeasible 'exato' instance) are logged and skipped."""
    done = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        prev = pd.read_csv(csv_path)
        done = set(zip(prev["algoritmo"], prev["instancia"], prev["corrida"]))
        print(f"  Reanudando: {len(done)} corridas ya completadas en {csv_path}")
    write_header = not done

    jobs = []
    for algo_name, (run_fn, folder) in ALGORITHMS.items():
        for instance_path, n in discover_instances(folder, min_n, max_n):
            jobs.append((algo_name, run_fn, instance_path, n))

    total_calls = len(jobs) * n_runs
    call_idx = 0

    print(f"{'=' * 70}")
    print(f"  {len(jobs)} combinaciones algoritmo x instancia, "
          f"{n_runs} corridas cada una -> {total_calls} corridas totales")
    print(f"{'=' * 70}")

    for algo_name, run_fn, instance_path, n in jobs:
        instancia = os.path.basename(instance_path)
        for i in range(n_runs):
            call_idx += 1
            if (algo_name, instancia, i + 1) in done:
                continue
            seed = i * 1000 + 42
            try:
                result = run_fn(instance_path, seed=seed, verbose=False)
            except ValueError as e:
                print(f"  [{call_idx}/{total_calls}] {algo_name} {instancia} "
                      f"corrida {i + 1}/{n_runs}... FALLO: {e}")
                continue

            row = {
                "algoritmo": algo_name,
                "instancia": instancia,
                "N": result["n"],
                "K": result["k"],
                "corrida": i + 1,
                "semilla": seed,
                "tiempo_s": result["elapsed_s"],
                "costo_determinista": result["cost"],
                "costo_esperado": result["expected_cost"],
                "iter_mejor": result["best_iteration"],
            }
            pd.DataFrame([row]).to_csv(
                csv_path, mode="a", header=write_header, index=False
            )
            write_header = False

            print(f"  [{call_idx}/{total_calls}] {algo_name} {instancia} "
                  f"corrida {i + 1}/{n_runs}... {result['elapsed_s']:.2f}s, "
                  f"esperado={result['expected_cost']:.2f}")

    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (algoritmo, instancia), aggregating the repetitions."""
    if df.empty:
        return pd.DataFrame(columns=[
            "algoritmo", "instancia", "N", "K", "corridas",
            "tiempo_prom_s", "tiempo_min_s", "tiempo_max_s",
            "esperado_prom", "esperado_min", "esperado_max", "esperado_std",
        ])

    grouped = df.groupby(["algoritmo", "instancia"], sort=False)
    summary = grouped.agg(
        N=("N", "first"),
        K=("K", "first"),
        corridas=("corrida", "count"),
        tiempo_prom_s=("tiempo_s", "mean"),
        tiempo_min_s=("tiempo_s", "min"),
        tiempo_max_s=("tiempo_s", "max"),
        esperado_prom=("costo_esperado", "mean"),
        esperado_min=("costo_esperado", "min"),
        esperado_max=("costo_esperado", "max"),
        esperado_std=("costo_esperado", "std"),
    ).reset_index()
    summary["esperado_std"] = summary["esperado_std"].fillna(0.0)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark en batch de los 4 solvers CaRS")
    parser.add_argument("--min-n", type=int, default=0, help="N minimo de instancia (inclusive)")
    parser.add_argument("--max-n", type=int, default=10000, help="N maximo de instancia (inclusive)")
    parser.add_argument("--runs", type=int, default=10, help="Corridas por (algoritmo, instancia)")
    parser.add_argument("--output", type=str, default=None, help="Ruta del .xlsx de salida")
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            "resultados", f"benchmark_N{args.min_n}-{args.max_n}.xlsx"
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    start = time.time()
    csv_path = os.path.splitext(output_path)[0] + ".csv"
    df_raw = run_batch(args.min_n, args.max_n, args.runs, csv_path)
    df_summary = summarize(df_raw)
    elapsed = time.time() - start

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="Resultados", index=False)
        df_summary.to_excel(writer, sheet_name="Resumen", index=False)

    print(f"\n{'=' * 70}")
    print(f"  LISTO en {elapsed:.1f}s -- {len(df_raw)} filas de resultados, "
          f"{len(df_summary)} combinaciones algoritmo x instancia")
    print(f"  Excel: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
