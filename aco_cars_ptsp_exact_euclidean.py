"""
ACO for the CaRS (Car Renter Salesman) problem -- Euclidean instances,
fixed version of aco_tesisv5.py.

Problem variant: "exato" (Silva, 2011 thesis, p.48, section 3.3 item 3):
    k = t -- ALL k available vehicles must be used at least once in the
    tour ("todos os carros devem ser obrigatoriamente alugados"). This is
    a deliberate, formally-defined CaRS variant chosen for this project
    (as opposed to the thesis's own default "sem repeticao" variant,
    k >= t, which does NOT require using every vehicle -- see p.44 item 5
    and Figure 7, where using a single vehicle for the whole tour is shown
    as a valid CaRS solution, degenerating to the classic TSP).

Bugs fixed relative to aco_tesisv5.py
--------------------------------------
1. `asignar_vehiculos_optimos()` only tried single-switch patterns
   ([0]*p + [1]*(n-p) and the reverse), and its range EXCLUDED the
   all-single-vehicle cases (p=0 and p=n_arcos) entirely. Verified
   numerically: wrong (suboptimal) result on 6/16 random test routes on
   BrasilRN16e, off by up to ~2.1% of route cost. Replaced here by a
   bitmask dynamic program that finds the TRUE global optimum over every
   possible multi-switch vehicle assignment, subject to the "exato"
   constraint that every vehicle appears at least once:

     dp[t][k][mask] = min cost of arcs 0..t, ending arc t with vehicle k,
                      having used exactly the set of vehicles in `mask`
                      (mask is a K-bit bitmask, bit k set means vehicle k
                      has appeared in some arc 0..t).

   Final answer = min_k dp[n_arcs-1][k][full_mask], full_mask = (1<<K)-1.
   Complexity O(N_arcos * K^2 * 2^K) -- trivial for K=2 (this instance).

2. Performance: `_precompute_costs()` built `edge_costs`/`return_costs`
   arrays once, but the vehicle-assignment step never used them -- it
   recomputed `np.linalg.norm` from scratch on every single call, for
   every ant, every iteration, every candidate. Fixed: the DP here always
   takes the precomputed (K,N,N)/(K,N) arrays directly.

3. Hardcoded to K=2 in the original; this version is K-agnostic.

4. ACS mechanics ported from the validated non-Euclidean pipeline
   (aco_cars_ptsp.py): q0 pseudorandom-proportional rule, candidate
   lists, local pheromone update, elitist global update with MMAS bounds
   (tau_min/tau_max), and stagnation detection + reset. The original
   tesisv5 ACS had none of these -- plain roulette selection and
   unbounded evaporation (which decays toward ~0 after ~200 iterations
   at rho=0.1, effectively losing edges permanently).

5. Or-opt added alongside 2-opt (the original only had 2-opt).

6. A brute-force cross-check (`assign_vehicles_bruteforce`) is included
   and run on the final best route to validate the DP independently, the
   same way expected_cost_ptsp_bruteforce() validated the PTSP formula.

Bug fixed (2026-09-04), relative to the thesis (Silva 2011, p.44 item 3,
p.47 worked example, p.54 RETURN_RATE_SECTION) -- same bug found and
fixed in aco_cars_ptsp_exact_noneuclidean.py (the non-Euclidean sibling of this file)
--------------------------------------------------------------------------
The thesis defines the return cost as d^k_ij: cost of returning vehicle
k, RENTED at city i, DELIVERED at city j, i!=j -- a function of BOTH
cities, not just j. This file's `return_costs` was built (in
build_instance) as a (K,N) array via
`return_costs[k][i] = ((2*ri)+(3*rj))/3 + dist[i][0]` -- rj and dist[i][0]
BOTH hardcoded to city 0 (the depot), so the array structurally could
only ever represent "return to the depot", never a vehicle's true rental
origin (which for every vehicle picked up mid-route is some OTHER city).
The DP then also never charged a return cost for the tour's LAST vehicle
at all. Both bugs only ever OMIT cost, so results came out below the
true optimum -- reproduced on BrasilRN16n.car (see aco_cars_ptsp_exact_noneuclidean.py
for the full analysis and the thesis's own worked numeric proof, p.47:
even the LAST car of a tour, which drives its final arc straight into
the depot, still pays to return to its own rental origin).

Fix: `return_costs` is now a full (K,N,N) array, generalizing the SAME
vector-plus-distance formula already used for edge_costs (thesis p.53,
VECTOR format: cost(i,j) = (2*v_i+3*v_j)/3 + dist(i,j)) to arbitrary j
instead of hardcoding j=0: `return_costs[k][i][j] = (2*ri+3*rj)/3 +
dist[i][j]`, ri=penalty[k][0][i], rj=penalty[k][0][j]. The DP
(`assign_vehicles_dp_all_used`) tracked, per (arc, vehicle, mask), the
city where the CURRENT vehicle's block started (its rental origin): a
switch paid return_costs[kp][origin_kp][src] (kp's real origin, not
column 0); the final answer added the still-open last vehicle's own
return_costs[k][origin_k][route[-1]] before declaring a winner.

Return cost set to 0 (2026-09, explicit request, on top of the fix above)
--------------------------------------------------------------------------
`assign_vehicles_dp_all_used` no longer charges return_costs at all --
switching vehicles is now free. Requested to explore the expected-cost
(PTSP) objective as the ACS's search criterion without conflating it
with a real cost term that expected_cost_ptsp() never modeled in the
first place (it only reads edge_costs, never return_costs). Since
switching is free, the DP no longer needs to track a block's origin --
state dropped back to dp[t][k][mask] (three dimensions, matching the
DP's shape from before the return-cost fix above). assign_vehicles_
bruteforce was updated the same way: plain sum of arc costs, no return
term.

Objective switched to expected cost (2026-09, same request as above)
--------------------------------------------------------------------------
`evaluate()` now returns (vehicles, cost, expected_cost): `cost` is still
the deterministic DP cost (kept only as a reference value, printed for
comparison), but the ACS search itself -- ant route selection each
iteration, 2-opt/Or-opt acceptance, the final local_search call, and the
MMAS global pheromone update (deposit/tau_max/tau_min) -- now all compare
and optimize `expected_cost` (expected_cost_ptsp()) instead of `cost`.
This mirrors the same change already made in aco_cars_ptsp_exact_noneuclidean.py.
Since return cost is 0 in the DP (see above) and expected_cost_ptsp()
never modeled return cost anyway, there is no remaining mismatch between
what the DP measures and what the PTSP formula measures.

Matplotlib plot removed, run() added (2026-09, batch benchmarking request)
--------------------------------------------------------------------------
The interactive `plt.show()` plot at the end of `__main__` blocked script
execution until the window was closed by hand -- incompatible with
running this file many times in a row from run_experiments.py. Removed
entirely (not gated -- there is no remaining plotting code in this file).
Added a `run(instance_path, seed, verbose)` function: loads an instance,
runs one ACS search, and returns a result dict (no printing unless
verbose=True) for programmatic/batch use. `__main__` now just calls
run() once and prints the result plus the existing DP/PTSP validations.

Usage:
    python aco_cars_ptsp_exact_euclidean.py
"""

import numpy as np
import random
import time
import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple


# =====================================================
# LECTURA DE INSTANCIA
# =====================================================

def read_car_instance(filename, verbose=True):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    N = 0
    K = 0
    coords = []

    for line in lines:
        if line.startswith("DIMENSION"):
            N = int(line.split(":")[1])
        elif line.startswith("CARS_NUMBER"):
            K = int(line.split(":")[1])

    idx = lines.index("NODE_COORD_SECTION") + 1
    while lines[idx] != "EDGE_WEIGHT_SECTION":
        p = lines[idx].split()
        coords.append((float(p[1]), float(p[2])))
        idx += 1

    idx += 1
    costs = []
    current = []
    while idx < len(lines):
        line = lines[idx]
        if line == "RETURN_RATE_SECTION":
            if current:
                costs.append(current)
            break
        vals = line.split()
        if len(vals) == 1 and vals[0].isdigit():
            if current:
                costs.append(current)
                current = []
        else:
            if all(v.replace('.', '', 1).isdigit() for v in vals):
                row = list(map(float, vals))
                current.append(row)
        idx += 1

    idx += 1
    penalty = []
    current = []
    while idx < len(lines):
        line = lines[idx]
        if line == "EOF" or line == "PROBABILITY_SECTION":
            if current:
                penalty.append(current)
            break
        vals = line.split()
        if len(vals) == 1 and vals[0].isdigit():
            if current:
                penalty.append(current)
                current = []
        else:
            if all(v.replace('.', '', 1).isdigit() for v in vals):
                row = list(map(float, vals))
                current.append(row)
        idx += 1

    probabilities = _parse_probabilities(lines, N, verbose=verbose)

    coords = np.array(coords)
    costs = np.array(costs)
    penalty = np.array(penalty)

    if verbose:
        print(f"DEBUG: N={N}, K={K}")
    return N, K, coords, costs, penalty, probabilities


def _parse_probabilities(lines, n, verbose=True):
    """Read PROBABILITY_SECTION section (P(j) presence probability per city).
    Default to 1.0 for all cities if the section is absent."""
    try:
        idx = lines.index("PROBABILITY_SECTION") + 1
    except ValueError:
        if verbose:
            print("  WARNING: PROBABILITY_SECTION section not found, using P=1.0 for all")
        return np.ones(n)

    probs = []
    for i in range(idx, min(idx + n, len(lines))):
        if lines[i] == "EOF":
            break
        probs.append(float(lines[i]))

    if len(probs) != n:
        raise ValueError(f"PROBABILITY_SECTION: expected {n} values, got {len(probs)}")

    return np.array(probs)


# =====================================================
# INSTANCIA PRECALCULADA
# =====================================================

@dataclass
class EuclideanInstance:
    N: int
    K: int
    coords: np.ndarray
    dist: np.ndarray        # (N, N)
    edge_costs: np.ndarray  # (K, N, N)
    return_costs: np.ndarray  # (K, N, N) -- return_costs[k][i][j]: cost of
                              # returning vehicle k, RENTED at city i,
                              # DELIVERED at city j (thesis d^k_ij, p.44)
    probabilities: np.ndarray  # (N,) -- P(j) presence probability per city (PTSP)


def build_instance(N, K, coords, costs, penalty, probabilities=None) -> EuclideanInstance:
    """Precompute all distance/cost arrays ONCE. Every downstream
    function (DP, ACS heuristic, 2-opt, Or-opt) must use these -- never
    recompute np.linalg.norm inside a hot loop."""
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])

    edge_costs = np.zeros((K, N, N))
    for k in range(K):
        for i in range(N):
            for j in range(N):
                if i != j:
                    vi = costs[k][0][i]
                    vj = costs[k][0][j]
                    edge_costs[k][i][j] = ((2 * vi) + (3 * vj)) / 3 + dist[i][j]

    # Same vector-plus-distance formula as edge_costs (thesis p.53, VECTOR
    # format), generalized to any (i,j) pair instead of hardcoding j=0 --
    # d^k_ij: return cost of a vehicle rented at i, delivered at j.
    return_costs = np.zeros((K, N, N))
    for k in range(K):
        for i in range(N):
            for j in range(N):
                if i != j:
                    ri = penalty[k][0][i]
                    rj = penalty[k][0][j]
                    return_costs[k][i][j] = ((2 * ri) + (3 * rj)) / 3 + dist[i][j]

    if probabilities is None:
        probabilities = np.ones(N)

    return EuclideanInstance(N, K, coords, dist, edge_costs, return_costs, probabilities)


# =====================================================
# ASIGNACION OPTIMA DE VEHICULOS -- DP con mascara de bits
# (variante "exato": todos los K vehiculos deben usarse)
# =====================================================

def assign_vehicles_dp_all_used(
    route: List[int],
    edge_costs: np.ndarray,
    return_costs: np.ndarray,
    n_vehicles: int,
) -> Tuple[Optional[List[int]], float]:
    """
    True optimal vehicle assignment for a fixed route, under the "exato"
    constraint (Silva 2011 thesis, p.48): every one of the n_vehicles
    vehicles must appear at least once, AND -- since "exato" is defined
    as a special case of "sem repeticao" -- once a vehicle is switched
    away from, it can never be picked up again later in the route. Each
    vehicle occupies exactly one contiguous block of arcs.

    RETURN COST TREATED AS 0 (2026-09, explicit request): this DP no
    longer charges return_costs[k][origin][dropoff] when a block closes
    -- switching vehicles is free. Requested to explore the
    expected-cost (PTSP) objective without conflating it with a cost
    term that formula itself never models (expected_cost_ptsp only
    reads edge_costs, never return_costs). `return_costs` is still
    accepted as a parameter (and EuclideanInstance.return_costs is still
    built from the .car file) purely so this can be reverted later; it
    is not used anywhere in this function.

    Since switching costs nothing, the origin of a vehicle's block no
    longer affects any future cost -- so, unlike the return-cost-aware
    version of this function (see project history), the DP does not
    need to track WHERE each block started. State drops back to three
    dimensions instead of four:

    dp[t][k][mask] = min cost of arcs 0..t, ending arc t with vehicle k,
                     mask = set of vehicles used in arcs 0..t (bit k set).

    A transition into vehicle k where bit k is ALREADY set in the target
    mask is only valid if the previous arc also used k (continuing the
    same block, no switch) -- never from a different kp, which would mean
    re-renting a vehicle already returned earlier.

    Returns (None, inf) if infeasible (fewer arcs than vehicles -- can't
    possibly use K distinct vehicles in fewer than K arcs).
    """
    n_arcs = len(route) - 1
    K = n_vehicles
    n_masks = 1 << K
    full_mask = n_masks - 1

    if n_arcs < K:
        return None, float('inf')

    INF = float('inf')
    dp = np.full((n_arcs, K, n_masks), INF)
    par_k = np.full((n_arcs, K, n_masks), -1, dtype=int)
    par_mask = np.full((n_arcs, K, n_masks), -1, dtype=int)

    for k in range(K):
        dp[0, k, 1 << k] = edge_costs[k, route[0], route[1]]

    for t in range(1, n_arcs):
        src, dst = route[t], route[t + 1]
        for k in range(K):
            bit = 1 << k
            ec = edge_costs[k, src, dst]
            for mask in range(n_masks):
                if not (mask & bit):
                    continue

                # Case 1: k was already used before -- only valid
                # continuation is staying on k (no re-entry after a
                # switch away).
                prev_cost = dp[t - 1, k, mask]
                if prev_cost != INF:
                    total = prev_cost + ec
                    if total < dp[t, k, mask]:
                        dp[t, k, mask] = total
                        par_k[t, k, mask] = k
                        par_mask[t, k, mask] = mask

                # Case 2: k is used here for the first time -- opens a
                # brand new block. Switching is free (return cost = 0),
                # so we just take whichever previously-active vehicle
                # kp had the cheapest cost so far.
                prev_mask = mask ^ bit
                for kp in range(K):
                    if kp == k:
                        continue
                    prev_cost = dp[t - 1, kp, prev_mask]
                    if prev_cost == INF:
                        continue
                    total = prev_cost + ec
                    if total < dp[t, k, mask]:
                        dp[t, k, mask] = total
                        par_k[t, k, mask] = kp
                        par_mask[t, k, mask] = prev_mask

    best_cost = INF
    best_k = -1
    for k in range(K):
        if dp[n_arcs - 1, k, full_mask] < best_cost:
            best_cost = dp[n_arcs - 1, k, full_mask]
            best_k = k

    if best_k == -1 or best_cost == INF:
        return None, float('inf')

    vehicles = [0] * n_arcs
    k, mask = best_k, full_mask
    vehicles[n_arcs - 1] = k
    for t in range(n_arcs - 1, 0, -1):
        kp = int(par_k[t, k, mask])
        pm = int(par_mask[t, k, mask])
        k, mask = kp, pm
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
    vehicle in exactly one contiguous block, never reused after a
    switch). Only for small n_arcs (validation).

    Return cost treated as 0 (see assign_vehicles_dp_all_used) -- cost
    per combo is just the sum of arc costs."""
    n_arcs = len(route) - 1
    K = n_vehicles

    best_cost = float('inf')
    best_vehicles = None

    for combo in itertools.product(range(K), repeat=n_arcs):
        if len(set(combo)) != K:
            continue  # doesn't use all vehicles -- invalid under "exato"

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
            continue  # reuses a vehicle already returned -- invalid under "sem repeticao"

        total = 0.0
        for t in range(n_arcs):
            i, j, k = route[t], route[t + 1], combo[t]
            total += edge_costs[k, i, j]

        if total < best_cost:
            best_cost = total
            best_vehicles = list(combo)

    return best_vehicles, best_cost


# =====================================================
# PTSP EXPECTED COST -- Vehicle-Aware (agnostic, no optimization)
# Idem aco_cars_ptsp_vehicle_aware.py / aco_cars_ptsp_exact_noneuclidean.py: la formula
# no toca coordenadas ni vectores por vehiculo -- solo lee inst.edge_costs,
# que ya trae la combinacion vector+distancia resuelta desde build_instance().
# =====================================================

def expected_cost_ptsp(
    route: List[int],
    vehicles: List[int],
    inst: EuclideanInstance,
) -> Tuple[float, float, float, float]:
    """
    Costo esperado bajo el modelo PTSP, usando la asignacion de vehiculos
    YA decidida para `route` (Solution.vehicles del ACO). Agnostica: no
    optimiza nada, solo mide.

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

    d = inst.edge_costs
    P = inst.probabilities[customers]

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
    inst: EuclideanInstance,
) -> float:
    """Enumera los 2^n escenarios de presencia/ausencia, calcula el costo
    REAL de cada uno (con la misma regla de vehiculo: llegada para cada
    tramo, salida para el regreso final al deposito) y pondera por
    probabilidad. Validacion cruzada de expected_cost_ptsp()."""
    depot = route[0]
    customers = route[1:-1]
    n = len(customers)

    if n == 0:
        return 0.0

    arr_vehicle = vehicles[0:n]
    dep_vehicle = vehicles[1:n + 1]
    d = inst.edge_costs
    P = inst.probabilities[customers]

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


def _run_ptsp_verification(route, vehicles, inst: EuclideanInstance, max_n_for_bruteforce: int = 20):
    """Valida expected_cost_ptsp() (formula cerrada) contra
    expected_cost_ptsp_bruteforce() (enumeracion exhaustiva)."""
    n = len(route) - 2

    print(f"\n{'=' * 70}")
    print("  VALIDACION -- costo esperado PTSP (formula vs fuerza bruta)")
    print(f"{'=' * 70}")

    if n > max_n_for_bruteforce:
        print(f"  SALTEADA (n={n} clientes -> 2^{n} escenarios, por encima "
              f"del limite de {max_n_for_bruteforce})")
        return

    formula_total, t1, t2, t3 = expected_cost_ptsp(route, vehicles, inst)
    bruteforce_total = expected_cost_ptsp_bruteforce(route, vehicles, inst)
    diff = abs(formula_total - bruteforce_total)
    tag = "MATCH" if diff < 1e-6 else f"MISMATCH (delta={diff:.6f})"

    print(f"  Formula (cerrada):    {formula_total:.6f}  (T1={t1:.2f} T2={t2:.2f} T3={t3:.2f})")
    print(f"  Fuerza bruta (2^{n}): {bruteforce_total:.6f}  ({2 ** n} escenarios)")
    print(f"  -> {tag}")
    print(f"{'=' * 70}")


def evaluate(route: List[int], inst: EuclideanInstance) -> Tuple[Optional[List[int]], float, float]:
    """Returns (vehicles, cost, expected_cost). `cost` is the deterministic
    DP cost (kept as a reference value only); `expected_cost` (PTSP,
    objective switched 2026-09) is what the ACS search actually optimizes."""
    vehicles, cost = assign_vehicles_dp_all_used(route, inst.edge_costs, inst.return_costs, inst.K)
    if vehicles is None:
        return None, float('inf'), float('inf')
    expected, _, _, _ = expected_cost_ptsp(route, vehicles, inst)
    return vehicles, cost, expected


# =====================================================
# BUSQUEDA LOCAL -- 2-opt
# =====================================================

def two_opt(route: List[int], inst: EuclideanInstance) -> Tuple[List[int], Optional[List[int]], float, float]:
    best_route = route[:]
    best_vehicles, best_cost, best_expected = evaluate(best_route, inst)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                new_vehicles, new_cost, new_expected = evaluate(new_route, inst)
                if new_expected < best_expected - 1e-9:
                    best_route, best_vehicles, best_cost, best_expected = new_route, new_vehicles, new_cost, new_expected
                    improved = True
                    break
            if improved:
                break

    return best_route, best_vehicles, best_cost, best_expected


# =====================================================
# BUSQUEDA LOCAL -- Or-opt
# =====================================================

def or_opt(route: List[int], inst: EuclideanInstance, segment_sizes=(1, 2, 3)) -> Tuple[List[int], Optional[List[int]], float, float]:
    best_route = route[:]
    best_vehicles, best_cost, best_expected = evaluate(best_route, inst)
    improved = True

    while improved:
        improved = False
        n = len(best_route)
        for seg_len in segment_sizes:
            for i in range(1, n - seg_len - 1):
                segment = best_route[i:i + seg_len]
                remaining = best_route[:i] + best_route[i + seg_len:]
                for j in range(1, len(remaining)):
                    if j == i:
                        continue
                    new_route = remaining[:j] + segment + remaining[j:]
                    if new_route[0] != 0 or new_route[-1] != 0:
                        continue
                    new_vehicles, new_cost, new_expected = evaluate(new_route, inst)
                    if new_expected < best_expected - 1e-9:
                        best_route, best_vehicles, best_cost, best_expected = new_route, new_vehicles, new_cost, new_expected
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_route, best_vehicles, best_cost, best_expected


def local_search(route: List[int], inst: EuclideanInstance) -> Tuple[List[int], Optional[List[int]], float, float]:
    current = route[:]
    improved = True
    while improved:
        improved = False
        new_route, new_vehicles, new_cost, new_expected = two_opt(current, inst)
        _, _, current_expected = evaluate(current, inst)
        if new_expected < current_expected - 1e-9:
            current = new_route
            improved = True
        new_route, new_vehicles, new_cost, new_expected = or_opt(current, inst)
        _, _, current_expected = evaluate(current, inst)
        if new_expected < current_expected - 1e-9:
            current = new_route
            improved = True
    vehicles, cost, expected = evaluate(current, inst)
    return current, vehicles, cost, expected


# =====================================================
# ANT COLONY SYSTEM (mecanicas de aco_cars_ptsp.py)
# =====================================================

@dataclass
class ACOConfig:
    alpha: float = 1.0
    beta: float = 2.5
    rho: float = 0.1
    q0: float = 0.5
    local_rho: float = 0.1
    n_ants: int = 30
    n_iterations: int = 200
    candidate_list_size: int = 10
    stagnation_limit: int = 50
    seed: int = 42


class ACS:
    def __init__(self, inst: EuclideanInstance, cfg: ACOConfig):
        self.inst = inst
        self.cfg = cfg
        self.N = inst.N
        self.K = inst.K

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.min_cost = np.min(inst.edge_costs, axis=0)  # (N,N) cheapest vehicle per arc, for heuristic only
        self.candidates = self._build_candidate_lists()
        self.heuristic = self._compute_heuristic()
        self.nn_cost = self._nearest_neighbor_cost()
        self.tau0 = 1.0 / (self.N * self.nn_cost) if self.nn_cost > 0 else 0.5
        self.pheromone = np.full((self.N, self.N), self.tau0)

        self.best_route: Optional[List[int]] = None
        self.best_vehicles: Optional[List[int]] = None
        self.best_cost = float('inf')
        self.best_expected = float('inf')
        self.history: List[float] = []
        self._last_improvement_iter = 0
        self._last_reset_iter = 0

    def _build_candidate_lists(self):
        size = min(self.cfg.candidate_list_size, self.N - 1)
        cl = np.zeros((self.N, size), dtype=int)
        for i in range(self.N):
            row = self.min_cost[i].copy()
            row[i] = np.inf
            cl[i] = np.argsort(row)[:size]
        return cl

    def _compute_heuristic(self):
        with np.errstate(divide="ignore"):
            eta = 1.0 / (self.min_cost + 1e-10)
        np.fill_diagonal(eta, 0.0)
        return eta

    def _nearest_neighbor_cost(self):
        route = [0]
        remaining = set(range(1, self.N))
        current = 0
        while remaining:
            nxt = min(remaining, key=lambda j: self.min_cost[current, j])
            route.append(nxt)
            remaining.remove(nxt)
            current = nxt
        route.append(0)
        _, _, expected = evaluate(route, self.inst)
        return expected if (expected != float('inf') and expected > 0) else 1.0

    def _select_next(self, current, unvisited_arr):
        if random.random() < self.cfg.q0:
            cl_avail = np.array([j for j in self.candidates[current] if j in unvisited_arr])
            pool = cl_avail if len(cl_avail) > 0 else unvisited_arr
            values = self.pheromone[current, pool] ** self.cfg.alpha * self.heuristic[current, pool] ** self.cfg.beta
            return int(pool[np.argmax(values)])

        probs = self.pheromone[current, unvisited_arr] ** self.cfg.alpha * self.heuristic[current, unvisited_arr] ** self.cfg.beta
        total = probs.sum()
        if total <= 1e-12:
            return int(np.random.choice(unvisited_arr))
        probs = probs / total
        return int(np.random.choice(unvisited_arr, p=probs))

    def _construct_route(self):
        route = [0]
        unvisited = set(range(1, self.N))
        current = 0
        while unvisited:
            unvisited_arr = np.array(list(unvisited))
            nxt = self._select_next(current, unvisited_arr)
            route.append(nxt)
            self.pheromone[current, nxt] = (
                (1 - self.cfg.local_rho) * self.pheromone[current, nxt]
                + self.cfg.local_rho * self.tau0
            )
            unvisited.remove(nxt)
            current = nxt
        route.append(0)
        return route

    def _global_pheromone_update(self):
        if self.best_route is None:
            return
        self.pheromone *= (1 - self.cfg.rho)
        deposit = self.cfg.rho / self.best_expected
        for t in range(len(self.best_route) - 1):
            self.pheromone[self.best_route[t], self.best_route[t + 1]] += deposit

        tau_max = 1.0 / (self.cfg.rho * self.best_expected)
        tau_min = tau_max / (2 * self.N)
        np.clip(self.pheromone, tau_min, tau_max, out=self.pheromone)

    def _is_stagnant(self, it):
        limit = self.cfg.stagnation_limit
        return (it - self._last_improvement_iter >= limit) and (it - self._last_reset_iter >= limit)

    def run(self, verbose=True):
        if verbose:
            print("\n" + "=" * 70)
            print("  ACS -- CaRS Euclidiano (con DP 'exato' + MMAS, objetivo: costo esperado PTSP)")
            print("=" * 70)
            print(f"  Nodos: {self.N}, Vehiculos: {self.K}")
            print(f"  Hormigas: {self.cfg.n_ants}, Iteraciones: {self.cfg.n_iterations}")
            print(f"  alpha={self.cfg.alpha} beta={self.cfg.beta} rho={self.cfg.rho} q0={self.cfg.q0}")
            print("=" * 70)

        start = time.time()
        for it in range(self.cfg.n_iterations):
            it_best_route, it_best_vehicles, it_best_cost, it_best_expected = None, None, float('inf'), float('inf')
            for _ in range(self.cfg.n_ants):
                route = self._construct_route()
                vehicles, cost, expected = evaluate(route, self.inst)
                if vehicles is not None and expected < it_best_expected:
                    it_best_route, it_best_vehicles, it_best_cost, it_best_expected = route, vehicles, cost, expected

            if it_best_route is not None and it_best_expected < self.best_expected:
                self.best_route, self.best_vehicles = it_best_route[:], it_best_vehicles[:]
                self.best_cost, self.best_expected = it_best_cost, it_best_expected
                self._last_improvement_iter = it

            self._global_pheromone_update()
            self.history.append(self.best_expected)

            if self._is_stagnant(it):
                if verbose:
                    print(f"  Iter {it + 1}: estancamiento detectado, reiniciando feromona")
                self.pheromone[:] = self.tau0
                self._last_reset_iter = it

            if verbose and (it % 20 == 0 or it == self.cfg.n_iterations - 1):
                print(f"  Iter {it + 1:4d}/{self.cfg.n_iterations}: Best (esperado) = {self.best_expected:.2f}  (deterministico = {self.best_cost:.2f})")

        if self.best_route is not None:
            self.best_route, self.best_vehicles, self.best_cost, self.best_expected = local_search(self.best_route, self.inst)

        if verbose:
            print(f"\n  Tiempo: {time.time() - start:.2f}s")

        return self.best_route, self.best_vehicles, self.best_cost, self.best_expected


# =====================================================
# API PROGRAMATICA -- para uso desde run_experiments.py
# =====================================================

def run(instance_path: str, seed: int = 42, verbose: bool = False) -> dict:
    """Carga instance_path, corre UNA busqueda ACS con la semilla dada, sin
    graficos ni ventanas. No corre las validaciones de fuerza bruta (serian
    demasiado lentas para correr cientos de veces en batch) -- esas quedan
    solo en __main__ para chequeos manuales puntuales.

    Devuelve:
        {
            "route": list[int], "vehicles": list[int],
            "cost": float,            # costo deterministico (referencia)
            "expected_cost": float,   # costo esperado PTSP (el objetivo)
            "elapsed_s": float,       # tiempo de la busqueda (ACS + local search)
            "best_iteration": int,    # iteracion en la que se hallo la mejor solucion
            "n": int, "k": int,
        }

    Lanza ValueError si la instancia es infactible para la variante 'exato'
    (menos arcos que vehiculos).
    """
    N, K, coords, costs, penalty, probabilities = read_car_instance(instance_path, verbose=verbose)
    inst = build_instance(N, K, coords, costs, penalty, probabilities)

    cfg = ACOConfig(alpha=1.0, beta=2.5, rho=0.1, q0=0.5, local_rho=0.1,
                     n_ants=30, n_iterations=200, candidate_list_size=10,
                     stagnation_limit=50, seed=seed)

    acs = ACS(inst, cfg)
    start = time.time()
    route, vehicles, cost, expected = acs.run(verbose=verbose)
    elapsed = time.time() - start

    if route is None or vehicles is None:
        raise ValueError(f"Sin solucion factible (variante 'exato') para {instance_path}")

    return {
        "route": route,
        "vehicles": vehicles,
        "cost": cost,
        "expected_cost": expected,
        "elapsed_s": elapsed,
        "best_iteration": acs._last_improvement_iter,
        "n": N,
        "k": K,
    }


# =====================================================
# EJECUCION PRINCIPAL
# =====================================================

if __name__ == "__main__":
    filename = "instances/euclidean/BrasilRJ14e.car"

    result = run(filename, seed=42, verbose=True)
    route, vehicles = result["route"], result["vehicles"]
    K = result["k"]

    print("\n" + "=" * 70)
    print("  RESULTADO (seleccionado por costo esperado PTSP)")
    print("=" * 70)
    print(f"  Costo esperado (PTSP):  {result['expected_cost']:.2f}")
    print(f"  Costo deterministico:   {result['cost']:.2f}  (referencia, no es el objetivo)")
    print(f"  Tiempo:                 {result['elapsed_s']:.2f}s")
    print(f"  Mejor iteracion:        {result['best_iteration']}")
    print(f"  Longitud ruta: {len(route)} nodos")
    print(f"  Vehiculos utilizados: {sorted(set(vehicles))}  (debe ser {list(range(K))}, variante 'exato')")

    print("\n  Ruta detallada:")
    for t in range(len(route) - 1):
        print(f"    {route[t]:3d} -> {route[t + 1]:3d} | Vehiculo {vehicles[t] + 1}")

    # -------------------------------------------------
    # Validaciones manuales (solo al correr este archivo directo, no en batch)
    # -------------------------------------------------
    N, K, coords, costs, penalty, probabilities = read_car_instance(filename, verbose=False)
    inst = build_instance(N, K, coords, costs, penalty, probabilities)

    n_arcs_final = len(route) - 1
    n_combinations = K ** n_arcs_final
    MAX_BRUTEFORCE_COMBINATIONS = 2_000_000
    print("\n" + "=" * 70)
    print("  VALIDACION -- DP 'exato' vs fuerza bruta")
    print("=" * 70)
    if n_combinations <= MAX_BRUTEFORCE_COMBINATIONS:
        bf_vehicles, bf_cost = assign_vehicles_bruteforce(
            route, inst.edge_costs, inst.return_costs, K
        )
        diff = abs(bf_cost - result["cost"])
        tag = "MATCH" if diff < 1e-6 else f"MISMATCH (delta={diff:.6f})"
        print(f"  DP:            {result['cost']:.6f}")
        print(f"  Fuerza bruta:  {bf_cost:.6f}  ({K}^{n_arcs_final} = {n_combinations} combinaciones)")
        print(f"  -> {tag}")
    else:
        print(f"  SALTEADA ({K}^{n_arcs_final} = {n_combinations} combinaciones, "
              f"por encima del limite de {MAX_BRUTEFORCE_COMBINATIONS})")
    print("=" * 70)

    _run_ptsp_verification(route, vehicles, inst)

    print("\n=== FIN ===")
