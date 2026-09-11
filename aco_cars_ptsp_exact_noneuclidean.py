"""
ACO for the CaRS (Car Renting Salesman) problem -- Non-Euclidean instances,
"exato" variant (Silva 2011 thesis, p.48, section 3.3 item 3): every
available vehicle must be used at least once (k = t), AND -- since "exato"
is a special case of "sem repeticao" -- once a vehicle is switched away
from, it can never be picked up again later in the route.

This file is aco_cars_ptsp_vehicle_aware.py with ONE change: the vehicle
assignment DP now enforces both constraints above, instead of the free
per-arc DP (assign_vehicles_dp) that could use as few as 1 vehicle and/or
revisit an already-returned vehicle. Everything else -- the vehicle-aware
PTSP expected-cost formula, the ACS mechanics (q0, candidate lists, MMAS
bounds, stagnation reset), 2-opt/Or-opt -- is unchanged.

The fix mirrors assign_vehicles_dp_all_used() in aco_cars_ptsp_exact_euclidean.py
(the Euclidean sibling of this file), adapted to this file's return_costs
shape (K, N, N).

Bug this fixes relative to assign_vehicles_dp (aco_cars_ptsp.py /
aco_cars_ptsp_vehicle_aware.py)
--------------------------------------------------------------------------
1. No guarantee every vehicle is used (could pick a single vehicle for
   the whole tour if that were cheapest) -- not what "exato" requires.
2. No memory of which vehicles were already used and abandoned: nothing
   stopped the DP from switching back to a vehicle already returned
   earlier (e.g. vehicle 0 -> vehicle 1 -> vehicle 0 again), which
   "sem repeticao" forbids. Verified with an adversarial synthetic case
   (see check scripts run in-session): an illegal "come back" pattern
   costing 6.5 was silently preferred by the old DP over the correct
   sem-repeticao-compliant answer, 54.5.

Second bug fixed (2026-09-04), relative to the thesis (Silva 2011,
"O Problema do Caixeiro Alugador", p.44 item 3, p.47 worked example,
p.54 RETURN_RATE_SECTION)
--------------------------------------------------------------------------
The thesis defines the return cost as d^k_ij: the cost of returning
vehicle k, RENTED at city i, and DELIVERED (dropped off) at city j, i!=j
-- return_costs[k][i][j] with i=rental origin, j=dropoff (same [row][col]
convention as edge_costs[k][i][j], both parsed identically as FULL_MATRIX
blocks). The previous version of this DP (and vehicle_aware.py) instead:
  1. Always assumed the return destination is column 0 (the depot),
     ignoring that only the FIRST vehicle of the tour has the depot as
     its rental origin -- every later vehicle is rented at whatever
     intermediate city it was first picked up at.
  2. Never charged a return cost for the LAST vehicle of the tour at
     all, since the DP's final answer was read straight off
     dp[n_arcs-1][k][full_mask] with nothing added afterwards.
Both bugs only ever OMIT cost, so the previous DP produced totals lower
than the true optimum -- reproduced on BrasilRN16n.car, where the "exato"
deterministic cost came out below the known-optimal value.

The thesis's own worked example (p.47, Figure 3-7) proves point 2: with
route F-A-B-E-C-D-F over 3 cars, the arcs cost 6, and ALL THREE cars pay
a return charge -- including car 3, the LAST one, which drives its final
arc straight into the depot F but still must return to C (where IT was
rented): "para o caso do carro 3, o retorno ao vertice C quando o carro
e entregue no vertice F custa duas unidades." Total: 6+1+2+2 = 11.

Fix: the DP state now tracks, per (arc, vehicle, mask), the city where
the CURRENT vehicle's block started (its rental origin). A switch pays
return_costs[kp][origin_kp][src] (kp's real origin, not column 0); the
final answer adds the still-open last vehicle's own
return_costs[k][origin_k][route[-1]] before declaring a winner.

State: dp[t][k][mask][origin] = min cost of arcs 0..t (return cost of the
still-open block NOT yet included), ending arc t with vehicle k, block
started at city `origin`, mask = set of vehicles used so far (bit k
set). A transition into k where bit k is already set in mask is only
valid if the previous arc ALSO used k (continuing the same block) --
never from a different vehicle, which would mean re-renting one already
returned.

Run directly:
    python aco_cars_ptsp_exact_noneuclidean.py
"""

import numpy as np
import random
import time
import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =====================================================
# DATA STRUCTURES
# =====================================================

@dataclass
class CaRSInstance:
    """Immutable problem data loaded from a .car file."""
    n_cities: int
    n_vehicles: int
    coords: np.ndarray        # (N, 2) — for visualization only
    edge_costs: np.ndarray    # (K, N, N) — travel cost per vehicle
    return_costs: np.ndarray  # (K, N, N) — return cost to depot per vehicle
    probabilities: np.ndarray  # (N,) — P(j) presence probability per city


@dataclass
class Solution:
    """A complete CaRS solution: route + per-arc vehicle assignment + cost."""
    route: List[int]
    vehicles: List[int]
    cost: float

    def copy(self) -> "Solution":
        return Solution(self.route[:], self.vehicles[:], self.cost)


@dataclass
class ACOConfig:
    """Hyperparameters for the Ant Colony System."""
    alpha: float = 1.0         # pheromone exponent
    beta: float = 2.5          # heuristic exponent
    rho: float = 0.1           # global evaporation rate
    q0: float = 0.5            # exploitation probability (ACS rule)
    local_rho: float = 0.1     # local pheromone decay after each step
    n_ants: int = 20
    n_iterations: int = 300
    candidate_list_size: int = 10
    stagnation_limit: int = 50
    seed: int = 42


# =====================================================
# INSTANCE I/O
# =====================================================

def read_instance(filename: str) -> CaRSInstance:
    """Read a non-Euclidean CaRS instance from .car format."""
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = _parse_int_field(lines, "DIMENSION")
    k = _parse_int_field(lines, "CARS_NUMBER")
    edge_costs = _parse_matrix_section(lines, "EDGE_WEIGHT_SECTION", "RETURN_RATE_SECTION")
    rc_end = "PROBABILITY_SECTION" if "PROBABILITY_SECTION" in lines else "EOF"
    return_costs = _parse_matrix_section(lines, "RETURN_RATE_SECTION", rc_end)
    coords = _circular_layout(n)
    probabilities = _parse_probabilities(lines, n)

    return CaRSInstance(
        n, k, coords, np.array(edge_costs), np.array(return_costs), probabilities
    )


def _parse_int_field(lines: List[str], key: str) -> int:
    for line in lines:
        if line.startswith(key):
            return int(line.split(":")[1].strip())
    raise ValueError(f"Field '{key}' not found in instance")


def _parse_matrix_section(
    lines: List[str], start_key: str, end_key: str
) -> List[np.ndarray]:
    try:
        idx_start = lines.index(start_key) + 1
    except ValueError:
        raise ValueError(f"Section '{start_key}' not found")

    idx_end = len(lines)
    for i in range(idx_start, len(lines)):
        if lines[i] == end_key or lines[i].startswith(end_key):
            idx_end = i
            break

    matrices: List[np.ndarray] = []
    current: List[List[float]] = []

    for i in range(idx_start, idx_end):
        line = lines[i]
        if line.isdigit():
            if current:
                matrices.append(np.array(current, dtype=float))
                current = []
        else:
            vals = line.split()
            if vals:
                current.append([float(v) for v in vals])

    if current:
        matrices.append(np.array(current, dtype=float))

    return matrices


def _circular_layout(n: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _parse_probabilities(lines: List[str], n: int) -> np.ndarray:
    try:
        idx = lines.index("PROBABILITY_SECTION") + 1
    except ValueError:
        return np.ones(n)

    probs = []
    for i in range(idx, min(idx + n, len(lines))):
        if lines[i] == "EOF":
            break
        probs.append(float(lines[i]))

    if len(probs) != n:
        raise ValueError(
            f"PROBABILITY_SECTION: expected {n} values, got {len(probs)}"
        )
    return np.array(probs)


# =====================================================
# VEHICLE ASSIGNMENT — Dynamic Programming ("exato" + "sem repeticao")
# =====================================================

def assign_vehicles_dp(
    route: List[int],
    edge_costs: np.ndarray,
    return_costs: np.ndarray,
    n_vehicles: int,
) -> Tuple[Optional[List[int]], float]:
    """
    Optimal vehicle assignment for a fixed route under the "exato"
    constraint: every one of the n_vehicles vehicles must be used at
    least once, AND (sem repeticao) once a vehicle is switched away from
    it can never be reused -- each vehicle occupies exactly one
    contiguous block of arcs.

    Return cost follows the thesis exactly (Silva 2011, p.44 item 3):
    d^k_ij = cost of returning vehicle k, RENTED at city i, DELIVERED at
    city j -- return_costs[k][i][j]. Every vehicle used pays this once,
    from its own rental origin to wherever it is dropped off, INCLUDING
    the last vehicle of the tour (p.47 worked example: the final car
    still pays to return to its origin, even though it drives its last
    arc straight into the depot).

    dp[t][k][mask][origin] = min cost of arcs 0..t (the still-open
    block's own return cost is NOT included yet), ending arc t with
    vehicle k, whose current block started at city `origin`, mask = set
    of vehicles used in arcs 0..t (bit k set).

    Returns (None, inf) if infeasible (fewer arcs than vehicles).
    """
    n_arcs = len(route) - 1
    K = n_vehicles
    n_masks = 1 << K
    full_mask = n_masks - 1
    N = edge_costs.shape[1]

    if n_arcs < K:
        return None, float('inf')

    INF = float('inf')
    dp = np.full((n_arcs, K, n_masks, N), INF)
    par_k = np.full((n_arcs, K, n_masks, N), -1, dtype=int)
    par_mask = np.full((n_arcs, K, n_masks, N), -1, dtype=int)
    par_origin = np.full((n_arcs, K, n_masks, N), -1, dtype=int)

    start_city = route[0]
    for k in range(K):
        dp[0, k, 1 << k, start_city] = edge_costs[k, route[0], route[1]]

    for t in range(1, n_arcs):
        src, dst = route[t], route[t + 1]
        for k in range(K):
            bit = 1 << k
            ec = edge_costs[k, src, dst]
            for mask in range(n_masks):
                if not (mask & bit):
                    continue

                # Case 1: k already used before -- only valid continuation
                # is staying on k (no re-entry after a switch away).
                # Its block's origin carries over unchanged.
                candidate = dp[t - 1, k, mask, :] + ec
                better = candidate < dp[t, k, mask, :]
                if np.any(better):
                    dp[t, k, mask, better] = candidate[better]
                    par_k[t, k, mask, better] = k
                    par_mask[t, k, mask, better] = mask
                    par_origin[t, k, mask, better] = np.nonzero(better)[0]

                # Case 2: k used here for the first time -- opens a brand
                # new block at city `src`. Whichever vehicle kp was
                # active closes its block right now and pays its OWN
                # return cost, from its real origin back to `src`.
                prev_mask = mask ^ bit
                for kp in range(K):
                    if kp == k:
                        continue
                    total_row = (
                        dp[t - 1, kp, prev_mask, :]
                        + return_costs[kp, :, src]
                        + ec
                    )
                    best_origin_kp = int(np.argmin(total_row))
                    best_total = total_row[best_origin_kp]
                    if best_total < dp[t, k, mask, src]:
                        dp[t, k, mask, src] = best_total
                        par_k[t, k, mask, src] = kp
                        par_mask[t, k, mask, src] = prev_mask
                        par_origin[t, k, mask, src] = best_origin_kp

    # Close the last (still-open) vehicle's block: it must also return
    # from its own rental origin to the city where the tour ends.
    end_city = route[-1]
    best_cost = INF
    best_k = -1
    best_origin = -1
    for k in range(K):
        row = dp[n_arcs - 1, k, full_mask, :]
        totals = row + return_costs[k, :, end_city]
        idx = int(np.argmin(totals))
        if totals[idx] < best_cost:
            best_cost = totals[idx]
            best_k = k
            best_origin = idx

    if best_k == -1 or best_cost == INF:
        return None, float('inf')

    vehicles = [0] * n_arcs
    k, mask, origin = best_k, full_mask, best_origin
    vehicles[n_arcs - 1] = k
    for t in range(n_arcs - 1, 0, -1):
        kp = int(par_k[t, k, mask, origin])
        pm = int(par_mask[t, k, mask, origin])
        po = int(par_origin[t, k, mask, origin])
        k, mask, origin = kp, pm, po
        vehicles[t - 1] = k

    return vehicles, float(best_cost)


def assign_vehicles_bruteforce(
    route: List[int],
    edge_costs: np.ndarray,
    return_costs: np.ndarray,
    n_vehicles: int,
) -> Tuple[Optional[List[int]], float]:
    """Literal enumeration of every K^n_arcs vehicle assignment, filtered
    to those using all K vehicles AND respecting "sem repeticao" (each
    vehicle in exactly one contiguous block). Only for small n_arcs.

    Cost per combo: arc costs + return_costs[k][origin][dropoff] for
    EVERY contiguous block (thesis d^k_ij), including the last one."""
    n_arcs = len(route) - 1
    K = n_vehicles

    best_cost = float('inf')
    best_vehicles = None

    for combo in itertools.product(range(K), repeat=n_arcs):
        if len(set(combo)) != K:
            continue

        retired = set()
        prev = combo[0]
        valid = True
        for k in combo[1:]:
            if k != prev:
                retired.add(prev)
                if k in retired:
                    valid = False
                    break
            prev = k
        if not valid:
            continue

        total = 0.0
        block_start_idx = 0
        for t in range(n_arcs):
            i, j, k = route[t], route[t + 1], combo[t]
            total += edge_costs[k, i, j]

            is_last_arc = t == n_arcs - 1
            switches_next = (not is_last_arc) and combo[t + 1] != k
            if switches_next or is_last_arc:
                total += return_costs[k, route[block_start_idx], route[t + 1]]
                block_start_idx = t + 1

        if total < best_cost:
            best_cost = total
            best_vehicles = list(combo)

    return best_vehicles, best_cost


# =====================================================
# ROUTE EVALUATION
# =====================================================

def evaluate(route: List[int], instance: CaRSInstance) -> Solution:
    """Evaluate a route: find optimal vehicles via DP + compute total cost."""
    vehicles, cost = assign_vehicles_dp(
        route, instance.edge_costs, instance.return_costs, instance.n_vehicles
    )
    if vehicles is None:
        return Solution(route, [], float('inf'))
    return Solution(route, vehicles, cost)


def verify_cost(
    sol: Solution, edge_costs: np.ndarray, return_costs: np.ndarray
) -> float:
    """Recompute cost by contiguous vehicle blocks, for manual verification.

    Each block pays edge costs for its arcs plus ONE return cost
    return_costs[k][origin][dropoff] (thesis d^k_ij) where origin is the
    city where that block started and dropoff is the city where it ends
    -- this applies to the LAST block too (return to its own origin,
    even though the tour physically ends at the depot)."""
    total = 0.0
    n_arcs = len(sol.route) - 1
    block_start_idx = 0

    for t in range(n_arcs):
        i, j, k = sol.route[t], sol.route[t + 1], sol.vehicles[t]
        total += edge_costs[k, i, j]

        is_last_arc = t == n_arcs - 1
        switches_next = (not is_last_arc) and sol.vehicles[t + 1] != k
        if switches_next or is_last_arc:
            origin_city = sol.route[block_start_idx]
            total += return_costs[k, origin_city, j]
            block_start_idx = t + 1

    return total


def cost_breakdown(
    sol: Solution, edge_costs: np.ndarray, return_costs: np.ndarray
) -> float:
    """Print detailed cost breakdown (per contiguous vehicle block) and
    return total. See verify_cost() for the block-return-cost model."""
    total = 0.0
    n_arcs = len(sol.route) - 1
    block_start_idx = 0

    print(f"\n{'=' * 60}")
    print("  COST BREAKDOWN")
    print(f"{'=' * 60}")

    for t in range(n_arcs):
        i, j, k = sol.route[t], sol.route[t + 1], sol.vehicles[t]
        ec = edge_costs[k, i, j]
        total += ec
        print(f"  Arc {t + 1:2d}: {i:2d} -> {j:2d} | Vehicle {k + 1} | Cost {ec:.2f}")

        is_last_arc = t == n_arcs - 1
        switches_next = (not is_last_arc) and sol.vehicles[t + 1] != k
        if switches_next or is_last_arc:
            origin_city = sol.route[block_start_idx]
            rc = return_costs[k, origin_city, j]
            total += rc
            print(
                f"         Return vehicle {k + 1} "
                f"at city {j} -> origin {origin_city} | Cost {rc:.2f}"
            )
            block_start_idx = t + 1

    print(f"{'─' * 60}")
    print(f"  TOTAL: {total:.2f}")
    print(f"{'=' * 60}")

    return total


def _run_dp_verification(sol: Solution, instance: CaRSInstance, max_combinations: int = 2_000_000):
    """Validate the 'exato'+'sem repeticao' DP against literal brute force."""
    n_arcs = len(sol.route) - 1
    n_combinations = instance.n_vehicles ** n_arcs
    print(f"\n{'=' * 60}")
    print("  VALIDACION -- DP 'exato' vs fuerza bruta")
    print(f"{'=' * 60}")
    if n_combinations > max_combinations:
        print(f"  SALTEADA ({instance.n_vehicles}^{n_arcs} = {n_combinations} "
              f"combinaciones, por encima del limite de {max_combinations})")
        return
    bf_vehicles, bf_cost = assign_vehicles_bruteforce(
        sol.route, instance.edge_costs, instance.return_costs, instance.n_vehicles
    )
    diff = abs(bf_cost - sol.cost)
    tag = "MATCH" if diff < 1e-6 else f"MISMATCH (delta={diff:.6f})"
    print(f"  DP:            {sol.cost:.6f}")
    print(f"  Fuerza bruta:  {bf_cost:.6f}  ({instance.n_vehicles}^{n_arcs} = {n_combinations} combinaciones)")
    print(f"  -> {tag}")
    print(f"{'=' * 60}")


# =====================================================
# LOCAL SEARCH — 2-opt
# =====================================================

def two_opt(
    sol: Solution, instance: CaRSInstance, max_passes: int = 50
) -> Solution:
    """2-opt local search with first-improvement strategy."""
    best = sol.copy()

    for _ in range(max_passes):
        improved = _two_opt_pass(best, instance)
        if improved is None:
            break
        best = improved

    return best


def _two_opt_pass(sol: Solution, instance: CaRSInstance) -> Optional[Solution]:
    """Try all 2-opt moves, return first improvement or None."""
    n = len(sol.route)

    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_route = (
                sol.route[:i]
                + sol.route[i : j + 1][::-1]
                + sol.route[j + 1 :]
            )
            new_sol = evaluate(new_route, instance)

            if new_sol.cost < sol.cost - 1e-10:
                return new_sol

    return None


# =====================================================
# LOCAL SEARCH — Or-opt
# =====================================================

def or_opt(
    sol: Solution,
    instance: CaRSInstance,
    segment_sizes: Tuple[int, ...] = (1, 2, 3),
) -> Solution:
    """Or-opt: relocate segments of 1, 2, or 3 consecutive cities."""
    best = sol.copy()
    improved = True

    while improved:
        improved = False
        result = _or_opt_pass(best, instance, segment_sizes)
        if result is not None:
            best = result
            improved = True

    return best


def _or_opt_pass(
    sol: Solution,
    instance: CaRSInstance,
    segment_sizes: Tuple[int, ...],
) -> Optional[Solution]:
    """Single or-opt pass: try all segment sizes, return first improvement."""
    n = len(sol.route)

    for seg_len in segment_sizes:
        for i in range(1, n - seg_len - 1):
            segment = sol.route[i : i + seg_len]
            remaining = sol.route[:i] + sol.route[i + seg_len :]

            for j in range(1, len(remaining)):
                if j == i:
                    continue

                new_route = remaining[:j] + segment + remaining[j:]

                if new_route[0] != 0 or new_route[-1] != 0:
                    continue

                new_sol = evaluate(new_route, instance)
                if new_sol.cost < sol.cost - 1e-10:
                    return new_sol

    return None


# =====================================================
# LOCAL SEARCH — Combined
# =====================================================

def local_search(sol: Solution, instance: CaRSInstance) -> Solution:
    """Apply 2-opt + or-opt iteratively until no improvement found."""
    current = sol.copy()
    improved = True

    while improved:
        improved = False

        after = two_opt(current, instance)
        if after.cost < current.cost - 1e-10:
            current = after
            improved = True

        after = or_opt(current, instance)
        if after.cost < current.cost - 1e-10:
            current = after
            improved = True

    return current


# =====================================================
# PTSP EXPECTED COST — Vehicle-Aware (agnostic, no optimization)
# =====================================================

def expected_cost_ptsp(
    route: List[int],
    vehicles: List[int],
    instance: CaRSInstance,
) -> Tuple[float, float, float, float]:
    """
    Compute the expected tour cost under the PTSP model, using the vehicle
    assignment ALREADY decided for `route` (Solution.vehicles, as found by
    assign_vehicles_dp during the ACO search). Agnostic: no optimization.

    T1 = sum_j d(v0, r_j)^{c_j} * P(r_j) * prod_{k<j} (1 - P(r_k))
    T2 = sum_i d(r_i, v0)^{c_i} * P(r_i) * prod_{k>i} (1 - P(r_k))
    T3 = sum_i sum_{j>i} d(r_i, r_j)^{c_j} * P(r_i) * P(r_j)
         * prod_{i<v<j} (1 - P(r_v))
    """
    depot = route[0]
    customers = route[1:-1]
    n = len(customers)

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    if len(vehicles) != len(route) - 1:
        raise ValueError(
            f"vehicles must have {len(route) - 1} entries (one per arc of "
            f"route), got {len(vehicles)}"
        )

    arr_vehicle = vehicles[0:n]
    dep_vehicle = vehicles[1:n + 1]

    d = instance.edge_costs
    P = instance.probabilities[customers]

    prefix = np.ones(n + 1)
    for j in range(n):
        prefix[j + 1] = prefix[j] * (1.0 - P[j])

    suffix = np.ones(n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] * (1.0 - P[i])

    term1 = 0.0
    for j in range(n):
        term1 += d[arr_vehicle[j], depot, customers[j]] * P[j] * prefix[j]

    term2 = 0.0
    for i in range(n):
        term2 += d[dep_vehicle[i], customers[i], depot] * P[i] * suffix[i + 1]

    term3 = 0.0
    for i in range(n):
        absent_between = 1.0
        for j in range(i + 1, n):
            term3 += (
                d[arr_vehicle[j], customers[i], customers[j]]
                * P[i] * P[j] * absent_between
            )
            absent_between *= (1.0 - P[j])

    return term1 + term2 + term3, term1, term2, term3


def expected_cost_ptsp_bruteforce(
    route: List[int],
    vehicles: List[int],
    instance: CaRSInstance,
) -> float:
    """Brute-force validation of expected_cost_ptsp() (see that function
    and aco_cars_ptsp_vehicle_aware.py for full documentation)."""
    depot = route[0]
    customers = route[1:-1]
    n = len(customers)

    if n == 0:
        return 0.0

    arr_vehicle = vehicles[0:n]
    dep_vehicle = vehicles[1:n + 1]
    d = instance.edge_costs
    P = instance.probabilities[customers]

    total_expected = 0.0

    for mask in range(1 << n):
        prob = 1.0
        for pos in range(n):
            present = (mask >> pos) & 1
            prob *= P[pos] if present else (1.0 - P[pos])
        if prob <= 0.0:
            continue

        cost = 0.0
        prev_pos = None
        for pos in range(n):
            if not (mask >> pos) & 1:
                continue
            city = customers[pos]
            if prev_pos is None:
                cost += d[arr_vehicle[pos], depot, city]
            else:
                cost += d[arr_vehicle[pos], customers[prev_pos], city]
            prev_pos = pos

        if prev_pos is not None:
            cost += d[dep_vehicle[prev_pos], customers[prev_pos], depot]

        total_expected += prob * cost

    return total_expected


def _run_ptsp_verification(
    sol: Solution, instance: CaRSInstance, max_n_for_bruteforce: int = 20
):
    """Validate expected_cost_ptsp() against exhaustive enumeration."""
    n = len(sol.route) - 2

    if n > max_n_for_bruteforce:
        print(
            f"\n  PTSP brute-force validation SKIPPED "
            f"(n={n} customers -> 2^{n} scenarios, over the "
            f"max_n_for_bruteforce={max_n_for_bruteforce} cap)."
        )
        return

    formula_total, _, _, _ = expected_cost_ptsp(sol.route, sol.vehicles, instance)
    bruteforce_total = expected_cost_ptsp_bruteforce(sol.route, sol.vehicles, instance)
    diff = abs(formula_total - bruteforce_total)
    tag = "MATCH" if diff < 1e-6 else f"MISMATCH (delta={diff:.6f})"

    print(f"\n  PTSP validation -- formula vs brute force (2^{n} = {2 ** n} scenarios):")
    print(f"    Formula (closed-form): {formula_total:.6f}")
    print(f"    Brute force (2^n):     {bruteforce_total:.6f}")
    print(f"    -> {tag}")


def _print_expected_cost(sol: Solution, instance: CaRSInstance):
    """Print the PTSP expected cost analysis for a solution (vehicle-aware)."""
    expected, t1, t2, t3 = expected_cost_ptsp(sol.route, sol.vehicles, instance)
    customers = sol.route[1:-1]
    probs = instance.probabilities[customers]
    expected_visited = np.sum(probs)

    print(f"\n{'=' * 60}")
    print("  PTSP EXPECTED COST — vehicle-aware (c_j / c_i, no optimization)")
    print(f"{'=' * 60}")
    print(f"  Deterministic cost:   {sol.cost:.2f}")
    print(
        f"  Expected cost (PTSP): {expected:.2f}  "
        f"(T1={t1:.2f}  T2={t2:.2f}  T3={t3:.2f})"
    )
    print(f"  Expected customers:   {expected_visited:.2f} / {len(customers)}")
    print(f"  Avg probability:      {np.mean(probs):.4f}")
    print(f"{'=' * 60}")


# =====================================================
# ANT COLONY SYSTEM — Helpers
# =====================================================

def build_candidate_lists(instance: CaRSInstance, cl_size: int) -> np.ndarray:
    """Pre-compute nearest-neighbor candidate lists using min edge cost."""
    N = instance.n_cities
    min_cost = np.min(instance.edge_costs, axis=0)
    size = min(cl_size, N - 1)
    cl = np.zeros((N, size), dtype=int)

    for i in range(N):
        row = min_cost[i].copy()
        row[i] = np.inf
        cl[i] = np.argsort(row)[:size]

    return cl


def compute_heuristic(instance: CaRSInstance) -> np.ndarray:
    """Heuristic information: inverse of min edge cost over all vehicles."""
    min_cost = np.min(instance.edge_costs, axis=0)
    with np.errstate(divide="ignore"):
        eta = 1.0 / (min_cost + 1e-10)
    np.fill_diagonal(eta, 0.0)
    return eta


def nearest_neighbor_cost(instance: CaRSInstance) -> float:
    """Greedy nearest-neighbor tour cost (for initial pheromone calibration)."""
    min_cost = np.min(instance.edge_costs, axis=0)
    route = [0]
    remaining = set(range(1, instance.n_cities))
    current = 0

    while remaining:
        nxt = min(remaining, key=lambda j: min_cost[current, j])
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt

    route.append(0)
    cost = evaluate(route, instance).cost
    return cost if cost != float('inf') else 1.0


# =====================================================
# ANT COLONY SYSTEM — Core
# =====================================================

class ACS:
    """
    Ant Colony System with:
      - Pseudorandom-proportional rule (q0)
      - Local pheromone update (exploration)
      - Global update on best-so-far (exploitation)
      - MMAS pheromone bounds (anti-stagnation)
      - Candidate lists (faster construction)
      - Stagnation detection + reset
      - Light 2-opt per iteration, full local search at end
    """

    def __init__(self, instance: CaRSInstance, config: ACOConfig):
        self.inst = instance
        self.cfg = config
        self.N = instance.n_cities
        self.K = instance.n_vehicles

        random.seed(config.seed)
        np.random.seed(config.seed)

        self.candidates = build_candidate_lists(instance, config.candidate_list_size)
        self.heuristic = compute_heuristic(instance)
        self.nn_cost = nearest_neighbor_cost(instance)
        self.tau0 = (
            1.0 / (self.N * self.nn_cost) if self.nn_cost > 0 else 0.5
        )
        self.pheromone = np.full((self.N, self.N), self.tau0)

        self.best: Optional[Solution] = None
        self.history: List[float] = []
        self._last_improvement_iter: int = 0
        self._last_reset_iter: int = 0

    def run(self, verbose: bool = True) -> Optional[Solution]:
        """Execute the full ACS algorithm."""
        if verbose:
            self._print_header()

        start = time.time()

        for it in range(self.cfg.n_iterations):
            it_best = self._run_iteration()

            if it_best and (self.best is None or it_best.cost < self.best.cost):
                self.best = it_best.copy()
                self._last_improvement_iter = it

            self._global_pheromone_update()
            self.history.append(
                self.best.cost if self.best else float("inf")
            )

            if self._is_stagnant(it):
                if verbose:
                    print(
                        f"  Iter {it + 1}: "
                        f"stagnation detected, resetting pheromone"
                    )
                self.pheromone[:] = self.tau0
                self._last_reset_iter = it

            if verbose and (it % 50 == 0 or it == self.cfg.n_iterations - 1):
                c = f"{self.best.cost:.2f}" if self.best else "N/A"
                print(f"  Iter {it + 1:4d}/{self.cfg.n_iterations}: Best = {c}")

        if self.best:
            self.best = local_search(self.best, self.inst)

        elapsed = time.time() - start
        if verbose:
            print(f"\n  Time: {elapsed:.2f}s")

        return self.best

    def _run_iteration(self) -> Optional[Solution]:
        """Run all ants, return the iteration best (no local search here)."""
        it_best: Optional[Solution] = None

        for _ in range(self.cfg.n_ants):
            route = self._construct_route()
            sol = evaluate(route, self.inst)

            if sol.cost == float('inf'):
                continue

            if it_best is None or sol.cost < it_best.cost:
                it_best = sol

        return it_best

    def _construct_route(self) -> List[int]:
        """Build one ant's route with ACS construction rule."""
        route = [0]
        unvisited = set(range(1, self.N))
        current = 0

        while unvisited:
            nxt = self._select_next(current, unvisited)
            route.append(nxt)

            self.pheromone[current, nxt] = (
                (1 - self.cfg.local_rho) * self.pheromone[current, nxt]
                + self.cfg.local_rho * self.tau0
            )

            unvisited.remove(nxt)
            current = nxt

        route.append(0)
        return route

    def _select_next(self, current: int, unvisited: set) -> int:
        """ACS pseudorandom-proportional selection."""
        cands = np.array(list(unvisited))

        if random.random() < self.cfg.q0:
            cl_avail = np.array(
                [j for j in self.candidates[current] if j in unvisited]
            )
            pool = cl_avail if len(cl_avail) > 0 else cands

            values = (
                self.pheromone[current, pool] ** self.cfg.alpha
                * self.heuristic[current, pool] ** self.cfg.beta
            )
            return int(pool[np.argmax(values)])

        probs = (
            self.pheromone[current, cands] ** self.cfg.alpha
            * self.heuristic[current, cands] ** self.cfg.beta
        )
        total = probs.sum()

        if total <= 1e-12:
            return int(np.random.choice(cands))

        probs /= total
        return int(np.random.choice(cands, p=probs))

    def _global_pheromone_update(self):
        """Evaporate all, deposit on best-so-far, clamp with MMAS bounds."""
        if self.best is None:
            return

        self.pheromone *= 1 - self.cfg.rho

        deposit = self.cfg.rho / self.best.cost
        route = self.best.route
        for t in range(len(route) - 1):
            self.pheromone[route[t], route[t + 1]] += deposit

        tau_max = 1.0 / (self.cfg.rho * self.best.cost)
        tau_min = tau_max / (2 * self.N)
        np.clip(self.pheromone, tau_min, tau_max, out=self.pheromone)

    def _is_stagnant(self, current_iter: int) -> bool:
        """True if no improvement for stagnation_limit iters since last reset."""
        limit = self.cfg.stagnation_limit
        since_improvement = current_iter - self._last_improvement_iter
        since_reset = current_iter - self._last_reset_iter

        return since_improvement >= limit and since_reset >= limit

    def _print_header(self):
        cfg = self.cfg
        print(f"\n{'=' * 60}")
        print("  ACS — CaRS (Non-Euclidean, 'exato' + 'sem repeticao')")
        print(f"{'=' * 60}")
        print(f"  Cities: {self.N}  Vehicles: {self.K}")
        print(f"  Ants: {cfg.n_ants}  Iterations: {cfg.n_iterations}")
        print(
            f"  alpha={cfg.alpha} beta={cfg.beta} "
            f"rho={cfg.rho} q0={cfg.q0}"
        )
        print(f"{'=' * 60}")


# =====================================================
# VISUALIZATION
# =====================================================

def plot_solution(sol: Solution, instance: CaRSInstance):
    """Show route plot and solution details side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    _draw_route(ax1, sol, instance)
    _draw_info(ax2, sol, instance)
    plt.tight_layout()
    plt.show()


def _draw_route(ax, sol: Solution, instance: CaRSInstance):
    """Draw the route on a circular layout with vehicle-colored arcs."""
    coords = instance.coords
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1",
        "#96CEB4", "#FFEAA7", "#DDA0DD",
    ]

    ax.scatter(coords[:, 0], coords[:, 1], c="blue", s=80)
    ax.scatter(
        coords[0, 0], coords[0, 1],
        c="red", s=150, marker="s", label="Depot",
    )

    for t in range(len(sol.route) - 1):
        i, j = sol.route[t], sol.route[t + 1]
        color = colors[sol.vehicles[t] % len(colors)]
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color=color, alpha=0.8, linewidth=3,
        )

    for idx, (x, y) in enumerate(coords):
        ax.text(
            x, y, str(idx), fontsize=12, fontweight="bold",
            bbox=dict(
                boxstyle="circle,pad=0.1", facecolor="white", alpha=0.8
            ),
        )

    patches = [
        Patch(facecolor=colors[k % len(colors)], label=f"Vehicle {k + 1}")
        for k in range(instance.n_vehicles)
    ]
    ax.legend(handles=patches, loc="upper right")
    ax.set_title(
        f"Route — Cost: {sol.cost:.2f}", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)


def _draw_info(ax, sol: Solution, instance: CaRSInstance):
    """Draw the solution summary panel."""
    ax.axis("off")

    n_switches = _count_switches(sol.vehicles)
    route_str = _format_route(sol.route)
    veh_str = " ".join(str(v + 1) for v in sol.vehicles)
    expected, _, _, _ = expected_cost_ptsp(sol.route, sol.vehicles, instance)

    text = (
        f"Route:\n  {route_str}\n\n"
        f"Vehicles: {veh_str}\n\n"
        f"Det. Cost:  {sol.cost:.2f}\n"
        f"PTSP Cost:  {expected:.2f}\n"
        f"Cities:     {instance.n_cities}\n"
        f"Arcs:       {len(sol.route) - 1}\n"
        f"Switches:   {n_switches}"
    )

    ax.text(
        0.5, 0.5, text,
        transform=ax.transAxes, fontsize=11,
        va="center", ha="center", fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8
        ),
    )
    ax.set_title("Solution Details", fontsize=14, fontweight="bold")


def _count_switches(vehicles: List[int]) -> int:
    """Count how many times the vehicle changes between consecutive arcs."""
    return sum(
        1 for t in range(len(vehicles) - 1)
        if vehicles[t] != vehicles[t + 1]
    )


def _format_route(route: List[int], max_per_line: int = 10) -> str:
    """Format a route into multi-line string for display."""
    parts = []
    for i in range(0, len(route), max_per_line):
        parts.append(
            " -> ".join(str(n) for n in route[i : i + max_per_line])
        )
    return "\n  ".join(parts)


# =====================================================
# EXPERIMENT RUNNER
# =====================================================

def run_experiment(
    filename: str,
    n_runs: int = 2,
    config: Optional[ACOConfig] = None,
) -> Optional[Solution]:
    """Run multiple ACS executions and return the best solution found."""
    if config is None:
        config = ACOConfig()

    instance = read_instance(filename)
    _print_instance_info(instance)

    best: Optional[Solution] = None
    costs: List[float] = []
    times: List[float] = []

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")

        run_cfg = ACOConfig(
            alpha=config.alpha,
            beta=config.beta,
            rho=config.rho,
            q0=config.q0,
            local_rho=config.local_rho,
            n_ants=config.n_ants,
            n_iterations=config.n_iterations,
            candidate_list_size=config.candidate_list_size,
            stagnation_limit=config.stagnation_limit,
            seed=run_idx * 1000 + 42,
        )

        acs = ACS(instance, run_cfg)
        t0 = time.time()
        sol = acs.run()
        elapsed = time.time() - t0
        times.append(elapsed)

        if sol is None:
            print("  No solution found")
            continue

        costs.append(sol.cost)
        if best is None or sol.cost < best.cost:
            best = sol.copy()
            print(f"  NEW BEST: {best.cost:.2f}")

    if best is None:
        print("\nNo valid solution found")
        return None

    _print_final_results(best, costs, times)
    _run_verification(best, instance)
    _run_dp_verification(best, instance)
    _print_expected_cost(best, instance)
    _run_ptsp_verification(best, instance)
    plot_solution(best, instance)

    return best


def _print_instance_info(instance: CaRSInstance):
    """Print instance summary."""
    print(f"{'=' * 60}")
    print(
        f"  Instance: {instance.n_cities} cities, "
        f"{instance.n_vehicles} vehicles"
    )
    print(f"  Edge costs:     {instance.edge_costs.shape}")
    print(f"  Return costs:   {instance.return_costs.shape}")
    print(f"  Probabilities:  {instance.probabilities.shape}")
    has_ptsp = not np.all(instance.probabilities == 1.0)
    print(f"  PTSP mode:      {'Yes' if has_ptsp else 'No (all P=1.0)'}")
    print(f"{'=' * 60}")


def _print_final_results(
    sol: Solution, costs: List[float], times: List[float]
):
    """Print final experiment results."""
    n_switches = _count_switches(sol.vehicles)

    print(f"\n{'=' * 60}")
    print("  FINAL RESULT")
    print(f"{'=' * 60}")
    print(f"  Cost: {sol.cost:.2f}")
    print(
        f"  Route: {len(sol.route)} nodes, "
        f"{n_switches} vehicle switches"
    )

    for t in range(len(sol.route) - 1):
        print(
            f"    {sol.route[t]:3d} -> {sol.route[t + 1]:3d} "
            f"| Vehicle {sol.vehicles[t] + 1}"
        )

    if costs:
        print(f"\n  Statistics over {len(costs)} runs:")
        print(f"    Best:     {min(costs):.2f}")
        print(f"    Worst:    {max(costs):.2f}")
        print(f"    Mean:     {np.mean(costs):.2f}")
        print(f"    Std:      {np.std(costs):.2f}")
        print(f"    Avg time: {np.mean(times):.2f}s")


def _run_verification(sol: Solution, instance: CaRSInstance):
    """Verify algorithm cost matches manual recalculation."""
    manual = verify_cost(sol, instance.edge_costs, instance.return_costs)
    diff = abs(sol.cost - manual)
    tag = "MATCH" if diff < 0.01 else f"MISMATCH (delta={diff:.2f})"
    print(
        f"\n  Verification: algo={sol.cost:.2f}  "
        f"manual={manual:.2f}  -> {tag}"
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    config = ACOConfig(
        alpha=1.0,
        beta=2.5,
        rho=0.1,
        q0=0.5,
        local_rho=0.1,
        n_ants=20,
        n_iterations=300,
        candidate_list_size=10,
        stagnation_limit=50,
    )

    run_experiment("instances/noneuclidean/BrasilRN16n.car", n_runs=2, config=config)
