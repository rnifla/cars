import random
import math
import numpy as np
import time
import os

# ------------------------------------------------------------
# Cambios 2026-09 (mismo cambio ya aplicado en
# aco_cars_ptsp_exact_noneuclidean.py / aco_cars_ptsp_exact_euclidean.py /
# alns_cars_noneuclidean.py):
#
# 1. Costo de retorno tratado como 0: CarRenterProblem.costo_retorno()
#    ahora siempre devuelve 0.0 -- cambiar de vehiculo pasa a ser
#    gratis. La formula real d^k_ij de la tesis (Silva 2011, p.44 item
#    3) queda conservada, sin usar, en _costo_retorno_real(), por si se
#    revierte este cambio mas adelante. Como TODAS las funciones de
#    costo determinista del archivo (costo_ruta_con_vehiculos, grafico)
#    pasan por costo_retorno(), este unico cambio las deja a todas
#    consistentes automaticamente.
#
# 2. Objetivo del ALNS cambiado a costo esperado PTSP:
#    ALNSSolver._evaluar_ruta() -- el criterio de aceptacion/rechazo
#    que usa toda la busqueda (2-opt local, destroy/repair, temple
#    simulado) -- ahora calcula la asignacion optima de vehiculos por
#    DP (para minimizar el costo de los arcos) y despues mide el costo
#    ESPERADO (calcular_esperanza_formula_con_vehiculos) sobre esa
#    asignacion; es ese valor el que se compara. El costo determinista
#    se sigue calculando y mostrando por separado (GUI, exportacion)
#    como referencia, nunca como criterio de busqueda.
#
# 3. GUI de Tkinter eliminada (2026-09, pedido de benchmarking en batch):
#    la clase CarsUnificadoGUI y sus graficos de matplotlib bloqueaban la
#    ejecucion esperando que se cerrara una ventana a mano -- incompatible
#    con correr este archivo muchas veces desde run_experiments.py. Se
#    agrego run(instance_path, seed, verbose): carga una instancia, corre
#    UNA busqueda ALNS con la semilla dada, sin graficos ni ventanas, y
#    devuelve un dict de resultado. CarRenterProblem y ALNSSolver (la
#    logica real) no cambiaron.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1. DEFINICIÓN DEL PROBLEMA CaRS (CON PROBABILIDADES)
# ------------------------------------------------------------
class CarRenterProblem:
    def __init__(self, num_ciudades, dist_matrix, edge_vectors, return_vectors):
        self.n = num_ciudades
        self.dist = dist_matrix
        self.edge = edge_vectors
        self.return_rate = return_vectors
        self.num_vehiculos = edge_vectors.shape[0]
        self.nombre_instancia = "desconocida"
        self.ruta_instancia = ""
        self.probabilidades = None
        self.coordenadas = None

    def costo_arco_sin_retorno(self, i, j, v):
        term = (self.edge[v][i] * 2 + self.edge[v][j] * 3) / 3.0
        return term + self.dist[i][j]

    def costo_retorno(self, ciudad_alquiler, ciudad_devolucion, v):
        """Costo de devolver el vehiculo v, ALQUILADO en ciudad_alquiler,
        ENTREGADO en ciudad_devolucion.

        RETURN COST TRATADO COMO 0 (2026-09, mismo cambio que
        aco_cars_ptsp_exact_noneuclidean.py / aco_cars_ptsp_exact_euclidean.py /
        alns_cars_noneuclidean.py): cambiar de vehiculo pasa a ser gratis.
        La formula real d^k_ij de la tesis (Silva 2011, p.44 item 3) queda
        en _costo_retorno_real(), sin usar en ningun lado del archivo,
        solo por si se revierte este cambio mas adelante."""
        return 0.0

    def _costo_retorno_real(self, ciudad_alquiler, ciudad_devolucion, v):
        """Formula real d^k_ij de la tesis (Silva 2011, p.44 item 3): costo
        de devolver el vehiculo v, ALQUILADO en ciudad_alquiler, ENTREGADO
        en ciudad_devolucion. Formula VECTOR (thesis p.53), igual que
        costo_arco_sin_retorno, generalizada a cualquier j (no solo el
        deposito). NO UTILIZADA actualmente -- ver costo_retorno()."""
        if ciudad_alquiler == ciudad_devolucion:
            return 0.0
        ri = self.return_rate[v][ciudad_alquiler]
        rj = self.return_rate[v][ciudad_devolucion]
        return (2 * ri + 3 * rj) / 3.0 + self.dist[ciudad_alquiler][ciudad_devolucion]

    def costo_ruta_con_vehiculos(self, ruta):
        arcos = [(ruta[i], ruta[i+1]) for i in range(len(ruta)-1)]
        L = len(arcos)
        if L == 0:
            return 0.0, [], set(), []

        cost_seg_sin_ret = [[[float('inf')] * self.num_vehiculos for _ in range(L)] for __ in range(L)]
        for v in range(self.num_vehiculos):
            pref = [0.0] * (L+1)
            for k in range(L):
                i, j = arcos[k]
                pref[k+1] = pref[k] + self.costo_arco_sin_retorno(i, j, v)
            for i in range(L):
                for j in range(i, L):
                    cost_seg_sin_ret[i][j][v] = pref[j+1] - pref[i]

        max_mask = 1 << self.num_vehiculos
        dp = [[float('inf')] * max_mask for _ in range(L+1)]
        dp[0][0] = 0.0
        decision = [[None] * max_mask for _ in range(L+1)]

        for pos in range(L):
            for mask in range(max_mask):
                if dp[pos][mask] == float('inf'):
                    continue
                for next_pos in range(pos+1, L+1):
                    for v in range(self.num_vehiculos):
                        if not (mask & (1 << v)):
                            ciudad_alquiler = ruta[pos]
                            if next_pos == L:
                                ciudad_devolucion = 0
                            else:
                                ciudad_devolucion = ruta[next_pos]
                            new_mask = mask | (1 << v)
                            costo_ret = self.costo_retorno(ciudad_alquiler, ciudad_devolucion, v)
                            costo_seg = cost_seg_sin_ret[pos][next_pos-1][v] + costo_ret
                            new_cost = dp[pos][mask] + costo_seg
                            if new_cost < dp[next_pos][new_mask]:
                                dp[next_pos][new_mask] = new_cost
                                decision[next_pos][new_mask] = (pos, mask, v, next_pos-1)

        # Variante "exato" (Silva 2011 thesis, p.48, section 3.3 item 3),
        # la misma que aco_cars_ptsp_exact_noneuclidean.py / aco_cars_ptsp_exact_euclidean.py /
        # alns_cars_noneuclidean.py: los num_vehiculos vehiculos deben usarse TODOS.
        full_mask = max_mask - 1
        if L < self.num_vehiculos or dp[L][full_mask] == float('inf'):
            return float('inf'), [], set(), []

        mejor_costo = dp[L][full_mask]
        mejor_mask = full_mask

        vehiculos_por_arco = [None] * L
        puntos_cambio = []

        pos = L
        mask = mejor_mask
        while pos > 0:
            prev_pos, prev_mask, v, end_seg = decision[pos][mask]
            for k in range(prev_pos, end_seg + 1):
                vehiculos_por_arco[k] = v
            if end_seg + 1 < L:
                ciudad_devolucion = ruta[end_seg + 1]
                if ciudad_devolucion != 0:
                    ciudad_alquiler = ruta[prev_pos]
                    puntos_cambio.append((ciudad_devolucion, v, ciudad_alquiler))
            pos = prev_pos
            mask = prev_mask

        if L > 0 and vehiculos_por_arco[-1] is not None:
            ultimo_v = vehiculos_por_arco[-1]
            for idx, v in enumerate(vehiculos_por_arco):
                if v == ultimo_v:
                    ciudad_alquiler = ruta[idx]
                    if ciudad_alquiler != 0:
                        puntos_cambio.append((0, ultimo_v, ciudad_alquiler))
                    break

        tipos_usados = set(v for v in vehiculos_por_arco if v is not None)
        return mejor_costo, vehiculos_por_arco, tipos_usados, puntos_cambio

    # ------------------------------------------------------------
    # COSTO ESPERADO -- FORMULA CERRADA O(n^2) (T1+T2+T3), vehicle-aware.
    # Espejo exacto de la de alns_cars_noneuclidean.py -- ver formula_corregida.html:
    # T1/T3 usan el vehiculo de LLEGADA al destino, T2 el de SALIDA del
    # origen. Se calcula UNA SOLA VEZ sobre la mejor ruta determinista
    # final; la fuerza bruta de abajo se usa solo para VALIDARLA.
    # ------------------------------------------------------------
    def calcular_esperanza_formula_con_vehiculos(self, ruta, vehiculos):
        if self.probabilidades is None:
            raise ValueError("No se cargaron probabilidades.")

        depot = ruta[0]
        customers = [c for c in ruta if c != 0]
        n = len(customers)

        if n == 0:
            return 0.0, 0.0, 0.0, 0.0

        L = len(ruta) - 1
        if len(vehiculos) != L:
            raise ValueError(
                f"vehiculos debe tener {L} elementos (uno por arco), tiene {len(vehiculos)}."
            )

        arr_vehicle = vehiculos[0:n]
        dep_vehicle = vehiculos[1:n + 1]
        P = [self.probabilidades[c] for c in customers]

        prefix = [1.0] * (n + 1)
        for j in range(n):
            prefix[j + 1] = prefix[j] * (1.0 - P[j])

        suffix = [1.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] * (1.0 - P[i])

        term1 = 0.0
        for j in range(n):
            term1 += self.costo_arco_sin_retorno(depot, customers[j], arr_vehicle[j]) * P[j] * prefix[j]

        term2 = 0.0
        for i in range(n):
            term2 += self.costo_arco_sin_retorno(customers[i], depot, dep_vehicle[i]) * P[i] * suffix[i + 1]

        term3 = 0.0
        for i in range(n):
            absent_between = 1.0
            for j in range(i + 1, n):
                term3 += (
                    self.costo_arco_sin_retorno(customers[i], customers[j], arr_vehicle[j])
                    * P[i] * P[j] * absent_between
                )
                absent_between *= (1.0 - P[j])

        return term1 + term2 + term3, term1, term2, term3

    # ------------------------------------------------------------
    # CALCULO EXACTO DEL COSTO ESPERADO (FUERZA BRUTA), vehicle-aware.
    # Espejo exacto de alns_cars_noneuclidean.py -- validacion de calcular_esperanza_
    # formula_con_vehiculos() por enumeracion exhaustiva 2^n.
    # ------------------------------------------------------------
    def calcular_esperanza_fuerza_bruta_con_vehiculos(self, ruta, vehiculos):
        if self.probabilidades is None:
            raise ValueError("No se cargaron probabilidades.")
        L = len(ruta) - 1
        if len(vehiculos) != L:
            raise ValueError(f"La lista de vehículos debe tener longitud {L} (arcos), tiene {len(vehiculos)}.")
        orden = [c for c in ruta if c != 0]
        m = len(orden)
        if m > 20:
            raise ValueError(f"Demasiados nodos ({m}) para fuerza bruta (máx 20).")
        p = self.probabilidades
        total = 0.0
        pos_in_ruta = {nodo: idx for idx, nodo in enumerate(ruta)}

        for mask in range(1 << m):
            S = []
            prob = 1.0
            for i, nodo in enumerate(orden):
                if mask & (1 << i):
                    S.append(nodo)
                    prob *= p[nodo]
                else:
                    prob *= (1 - p[nodo])
            subtour = [0] + S + [0]
            costo = 0.0
            for idx in range(len(subtour) - 1):
                a = subtour[idx]
                b = subtour[idx + 1]
                if a == 0 and b == 0:
                    continue
                if b == 0:
                    # Termino 2 (ultimo presente -> deposito): vehiculo de
                    # SALIDA del origen a (formula_corregida.html, c_i).
                    pos_a = pos_in_ruta[a]
                    v = vehiculos[pos_a] if pos_a < L else vehiculos[-1]
                else:
                    # Terminos 1 y 3: vehiculo de LLEGADA al destino b
                    # (formula_corregida.html, seccion 9, c_j).
                    pos_b = pos_in_ruta[b]
                    v = vehiculos[pos_b - 1]
                costo += self.costo_arco_sin_retorno(a, b, v)
            total += prob * costo
        return total

    def costo_variable_ruta(self, segmento):
        if len(segmento) != 2:
            raise ValueError("costo_variable_ruta solo soporta segmentos de 2 ciudades")
        i, j = segmento
        return self.costo_arco_sin_retorno(i, j, 0)


# ------------------------------------------------------------
# 2. CARGA DE INSTANCIA (CON PROBABILITY_SECTION CORREGIDA)
# ------------------------------------------------------------
def cargar_instancia_cars(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        lineas = [line.rstrip('\n') for line in f]

    n = None
    n_vehiculos = None
    coordenadas = []
    edge_vectors = None
    return_vectors = None
    probabilidades = None
    nombre_instancia = os.path.basename(ruta_archivo)

    i = 0
    total = len(lineas)
    estado = None

    while i < total:
        linea = lineas[i].strip()
        i += 1
        if not linea:
            continue

        if linea.startswith('NAME'):
            nombre_instancia = linea.split(':')[1].strip()
        elif linea.startswith('DIMENSION'):
            n = int(linea.split(':')[1].strip())
        elif linea.startswith('CARS_NUMBER'):
            n_vehiculos = int(linea.split(':')[1].strip())
        elif linea.startswith('NODE_COORD_SECTION'):
            estado = 'COORDS'
            continue
        elif linea.startswith('EDGE_WEIGHT_SECTION'):
            estado = 'EDGE'
            edge_vectors = [None] * n_vehiculos
            continue
        elif linea.startswith('RETURN_RATE_SECTION'):
            estado = 'RETURN'
            return_vectors = [None] * n_vehiculos
            continue
        elif linea.startswith('PROBABILITY_SECTION'):
            estado = 'PROBABILITY_SECTION'
            probabilidades = []
            continue
        elif linea.startswith('EOF'):
            break

        if estado == 'COORDS':
            if len(coordenadas) < n:
                partes = linea.split()
                if len(partes) >= 3:
                    x = float(partes[1])
                    y = float(partes[2])
                    coordenadas.append((x, y))
            else:
                estado = None
                continue

        elif estado in ('EDGE', 'RETURN'):
            if not linea.isdigit():
                continue
            idx = int(linea)
            while i < total and lineas[i].strip() == '':
                i += 1
            if i >= total:
                break
            val_line = lineas[i].strip()
            i += 1
            valores = list(map(int, val_line.split()))
            if len(valores) != n:
                raise ValueError(f"Se esperaban {n} valores para vehículo {idx}, se obtuvieron {len(valores)}")
            if estado == 'EDGE':
                edge_vectors[idx] = valores
            else:
                return_vectors[idx] = valores
            continue

        # ========== CORRECCIÓN AQUÍ ==========
        elif estado == 'PROBABILITY_SECTION':
            # Lee todos los números que aparecen en la línea (pueden ser varios)
            for val in linea.split():
                try:
                    probabilidades.append(float(val))
                except ValueError:
                    pass
            if len(probabilidades) == n:
                estado = None
            continue
        # =====================================

    if n is None or n_vehiculos is None:
        raise ValueError("No se encontraron DIMENSION o CARS_NUMBER")
    if any(v is None for v in edge_vectors):
        raise ValueError("EDGE_WEIGHT_SECTION incompleta")
    if any(v is None for v in return_vectors):
        raise ValueError("RETURN_RATE_SECTION incompleta")
    if len(coordenadas) != n:
        coordenadas = [(0,0) for _ in range(n)]
    if probabilidades is None:
        probabilidades = [1.0] * n
        print("Advertencia: No se encontró PROBABILITY_SECTION. Se asume 1.0 para todas.")
    elif len(probabilidades) != n:
        raise ValueError(f"Se esperaban {n} probabilidades, se obtuvieron {len(probabilidades)}")

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        xi, yi = coordenadas[i]
        for j in range(n):
            if i != j:
                xj, yj = coordenadas[j]
                dx = xi - xj
                dy = yi - yj
                dist_matrix[i][j] = math.sqrt(dx*dx + dy*dy)

    edge_arr = np.array(edge_vectors, dtype=float)
    return_arr = np.array(return_vectors, dtype=float)

    return dist_matrix, edge_arr, return_arr, n, n_vehiculos, coordenadas, nombre_instancia, probabilidades


# ------------------------------------------------------------
# 5. ALGORITMO ALNS (OBJETIVO: ESPERANZA)
# ------------------------------------------------------------
class ALNSSolver:
    def __init__(self, problema, max_iter=100, temp_inicial=100,
                 cooling=0.99, porc_destruccion=0.3, verbose=False,
                 usar_estanc_iter=True, max_estanc_iter=200,
                 usar_estanc_tiempo=True, max_estanc_tiempo=60.0):
        self.problema = problema
        self.max_iter = max_iter
        self.temp = temp_inicial
        self.cooling = cooling
        self.porc_destruccion = porc_destruccion
        self.verbose = verbose
        self.destroy_ops = [
            self.destruir_random,
            self.destruir_worst,
            self.destruir_shaw,
            self.destruir_route_sequence
        ]
        self.repair_ops = [
            self.reconstruir_greedy,
            self.reconstruir_regret2,
            self.reconstruir_random,
            self.reconstruir_greedy_noise
        ]
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)
        self.destroy_usage = [0] * len(self.destroy_ops)
        self.repair_usage = [0] * len(self.repair_ops)
        self.destroy_scores = [0] * len(self.destroy_ops)
        self.repair_scores = [0] * len(self.repair_ops)
        self.total_destroy_usage = [0] * len(self.destroy_ops)
        self.total_repair_usage = [0] * len(self.repair_ops)
        self.initial_destroy_weights = self.destroy_weights.copy()
        self.initial_repair_weights = self.repair_weights.copy()
        self.segment_size = 100
        self.iter_count = 0
        self.historial_costos = []
        self.mejor_ruta = None
        self.mejor_costo = float('inf')
        self.best_iteration = 0
        self.tiempo_mejor = 0.0

        # Costo esperado (PTSP) de la mejor ruta determinista final --
        # se llenan UNA SOLA VEZ, al terminar la busqueda (ver
        # _calcular_expected_y_validar), nunca durante las iteraciones.
        self.mejor_costo_esperado = None
        self.mejor_costo_esperado_fb = None
        self.mejor_costo_esperado_diff = None
        self.mejor_costo_esperado_fb_skip_reason = None

        self.callback_mejora = None
        self.callback_progreso = None
        self.pausado = False

        self.usar_estanc_iter = usar_estanc_iter
        self.max_estanc_iter = max_estanc_iter
        self.usar_estanc_tiempo = usar_estanc_tiempo
        self.max_estanc_tiempo = max_estanc_tiempo

    def _evaluar_ruta(self, ruta):
        """Costo ESPERADO (PTSP) de la ruta -- objetivo switched a costo
        esperado (2026-09, mismo cambio que
        aco_cars_ptsp_exact_noneuclidean.py / aco_cars_ptsp_exact_euclidean.py /
        alns_cars_noneuclidean.py): este es el valor que compara/acepta
        toda la busqueda del ALNS (2-opt local, destroy/repair, temple
        simulado). Se calcula la asignacion de vehiculos que minimiza el
        costo de los arcos (DP), y sobre ESA asignacion se mide el costo
        esperado -- se llama en cada evaluacion de la busqueda, a
        diferencia de la validacion cruzada contra fuerza bruta, que
        sigue calculandose UNA SOLA VEZ al final (ver
        _calcular_expected_y_validar)."""
        if len(ruta) < 2:
            return 0.0
        _, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(ruta)
        if not vehiculos:
            return float('inf')
        esperado, _, _, _ = self.problema.calcular_esperanza_formula_con_vehiculos(ruta, vehiculos)
        return esperado

    def destruir_random(self, ruta):
        clientes = ruta[1:-1]
        n = max(1, int(len(clientes) * self.porc_destruccion))
        indices = random.sample(range(len(clientes)), n)
        eliminados = [clientes[i] for i in indices]
        nueva = [0] + [c for i, c in enumerate(clientes) if i not in indices] + [0]
        return nueva, eliminados

    def destruir_worst(self, ruta):
        clientes = ruta[1:-1]
        n = max(1, int(len(clientes) * self.porc_destruccion))
        contribuciones = []
        for idx in range(len(clientes)):
            pos = idx + 1
            costo_con = (self.problema.costo_variable_ruta([ruta[pos-1], ruta[pos]]) +
                         self.problema.costo_variable_ruta([ruta[pos], ruta[pos+1]]))
            costo_sin = self.problema.costo_variable_ruta([ruta[pos-1], ruta[pos+1]])
            contribuciones.append((idx, costo_con - costo_sin))
        contribuciones.sort(key=lambda x: x[1], reverse=True)
        indices = [c[0] for c in contribuciones[:n]]
        eliminados = [clientes[i] for i in indices]
        nueva = [0] + [c for i, c in enumerate(clientes) if i not in indices] + [0]
        return nueva, eliminados

    def destruir_shaw(self, ruta):
        clientes = ruta[1:-1]
        if not clientes:
            return self.destruir_random(ruta)
        n = max(1, int(len(clientes) * self.porc_destruccion))
        seed_idx = random.randrange(len(clientes))
        seed = clientes[seed_idx]
        eliminados = [seed]
        indices_elim = [seed_idx]
        distancias = [(i, self.problema.dist[seed][c]) for i, c in enumerate(clientes) if i != seed_idx]
        distancias.sort(key=lambda x: x[1])
        for i, _ in distancias:
            if len(eliminados) >= n:
                break
            if i not in indices_elim:
                eliminados.append(clientes[i])
                indices_elim.append(i)
        nueva = [0] + [c for i, c in enumerate(clientes) if i not in indices_elim] + [0]
        return nueva, eliminados

    def destruir_route_sequence(self, ruta):
        clientes = ruta[1:-1]
        if not clientes:
            return self.destruir_random(ruta)
        n = max(1, int(len(clientes) * self.porc_destruccion))
        start = random.randrange(len(clientes))
        indices = []
        for k in range(n):
            idx = (start + k) % len(clientes)
            if idx not in indices:
                indices.append(idx)
            else:
                break
        eliminados = [clientes[i] for i in indices]
        nueva = [0] + [c for i, c in enumerate(clientes) if i not in indices] + [0]
        return nueva, eliminados

    def reconstruir_greedy(self, ruta_parcial, eliminados):
        ruta = ruta_parcial.copy()
        random.shuffle(eliminados)
        for cliente in eliminados:
            mejor_pos = 1
            mejor_inc = float('inf')
            for pos in range(1, len(ruta)):
                if pos == 1:
                    inc = (self.problema.costo_variable_ruta([ruta[0], cliente]) +
                           self.problema.costo_variable_ruta([cliente, ruta[1]]) -
                           self.problema.costo_variable_ruta([ruta[0], ruta[1]]))
                elif pos == len(ruta)-1:
                    inc = (self.problema.costo_variable_ruta([ruta[-2], cliente]) +
                           self.problema.costo_variable_ruta([cliente, ruta[-1]]) -
                           self.problema.costo_variable_ruta([ruta[-2], ruta[-1]]))
                else:
                    inc = (self.problema.costo_variable_ruta([ruta[pos-1], cliente]) +
                           self.problema.costo_variable_ruta([cliente, ruta[pos]]) -
                           self.problema.costo_variable_ruta([ruta[pos-1], ruta[pos]]))
                if inc < mejor_inc:
                    mejor_inc = inc
                    mejor_pos = pos
            ruta.insert(mejor_pos, cliente)
        return ruta

    def reconstruir_regret2(self, ruta_parcial, eliminados):
        ruta = ruta_parcial.copy()
        while eliminados:
            best_regret = -1
            best_cliente = None
            best_pos = None
            for cliente in eliminados:
                mejores = []
                for pos in range(1, len(ruta)):
                    if pos == 1:
                        inc = (self.problema.costo_variable_ruta([ruta[0], cliente]) +
                               self.problema.costo_variable_ruta([cliente, ruta[1]]) -
                               self.problema.costo_variable_ruta([ruta[0], ruta[1]]))
                    elif pos == len(ruta)-1:
                        inc = (self.problema.costo_variable_ruta([ruta[-2], cliente]) +
                               self.problema.costo_variable_ruta([cliente, ruta[-1]]) -
                               self.problema.costo_variable_ruta([ruta[-2], ruta[-1]]))
                    else:
                        inc = (self.problema.costo_variable_ruta([ruta[pos-1], cliente]) +
                               self.problema.costo_variable_ruta([cliente, ruta[pos]]) -
                               self.problema.costo_variable_ruta([ruta[pos-1], ruta[pos]]))
                    mejores.append((pos, inc))
                mejores.sort(key=lambda x: x[1])
                regret = mejores[1][1] - mejores[0][1] if len(mejores) >= 2 else mejores[0][1]
                if regret > best_regret:
                    best_regret = regret
                    best_cliente = cliente
                    best_pos = mejores[0][0]
            ruta.insert(best_pos, best_cliente)
            eliminados.remove(best_cliente)
        return ruta

    def reconstruir_random(self, ruta_parcial, eliminados):
        ruta = ruta_parcial.copy()
        random.shuffle(eliminados)
        for cliente in eliminados:
            pos = random.randint(1, len(ruta)-1)
            ruta.insert(pos, cliente)
        return ruta

    def reconstruir_greedy_noise(self, ruta_parcial, eliminados):
        ruta = ruta_parcial.copy()
        random.shuffle(eliminados)
        max_dist = np.max(self.problema.dist)
        mu = 0.1
        for cliente in eliminados:
            mejor_pos = 1
            mejor_inc = float('inf')
            for pos in range(1, len(ruta)):
                if pos == 1:
                    inc = (self.problema.costo_variable_ruta([ruta[0], cliente]) +
                           self.problema.costo_variable_ruta([cliente, ruta[1]]) -
                           self.problema.costo_variable_ruta([ruta[0], ruta[1]]))
                elif pos == len(ruta)-1:
                    inc = (self.problema.costo_variable_ruta([ruta[-2], cliente]) +
                           self.problema.costo_variable_ruta([cliente, ruta[-1]]) -
                           self.problema.costo_variable_ruta([ruta[-2], ruta[-1]]))
                else:
                    inc = (self.problema.costo_variable_ruta([ruta[pos-1], cliente]) +
                           self.problema.costo_variable_ruta([cliente, ruta[pos]]) -
                           self.problema.costo_variable_ruta([ruta[pos-1], ruta[pos]]))
                noise = random.uniform(-max_dist*mu, max_dist*mu)
                if inc + noise < mejor_inc:
                    mejor_inc = inc + noise
                    mejor_pos = pos
            ruta.insert(mejor_pos, cliente)
        return ruta

    def seleccionar_operador(self, pesos):
        total = sum(pesos)
        r = random.uniform(0, total)
        acum = 0
        for i, p in enumerate(pesos):
            acum += p
            if r <= acum:
                return i
        return len(pesos)-1

    def actualizar_pesos(self):
        for i in range(len(self.destroy_ops)):
            if self.destroy_usage[i] > 0:
                tasa = self.destroy_scores[i] / self.destroy_usage[i]
                self.destroy_weights[i] = 0.8 * self.destroy_weights[i] + 0.2 * tasa
        for i in range(len(self.repair_ops)):
            if self.repair_usage[i] > 0:
                tasa = self.repair_scores[i] / self.repair_usage[i]
                self.repair_weights[i] = 0.8 * self.repair_weights[i] + 0.2 * tasa
        self.destroy_usage = [0] * len(self.destroy_ops)
        self.repair_usage = [0] * len(self.repair_ops)
        self.destroy_scores = [0] * len(self.destroy_ops)
        self.repair_scores = [0] * len(self.repair_ops)

    def aceptar(self, costo_nuevo, costo_actual):
        if costo_nuevo < costo_actual:
            return True
        prob = math.exp(-(costo_nuevo - costo_actual) / self.temp)
        return random.random() < prob

    def busqueda_local_2opt(self, ruta):
        mejor_ruta = ruta.copy()
        mejor_costo = self._evaluar_ruta(mejor_ruta)
        mejorado = True
        while mejorado:
            mejorado = False
            for i in range(1, len(mejor_ruta)-2):
                for j in range(i+1, len(mejor_ruta)-1):
                    nueva = mejor_ruta[:i] + mejor_ruta[i:j+1][::-1] + mejor_ruta[j+1:]
                    nuevo_costo = self._evaluar_ruta(nueva)
                    if nuevo_costo < mejor_costo:
                        mejor_ruta = nueva
                        mejor_costo = nuevo_costo
                        mejorado = True
                        break
                if mejorado:
                    break
        return mejor_ruta

    def solucion_inicial_aleatoria(self):
        ciudades = list(range(1, self.problema.n))
        random.shuffle(ciudades)
        return [0] + ciudades + [0]

    def _calcular_expected_y_validar(self, max_n_para_fuerza_bruta=20):
        """Paso final, UNA SOLA VEZ: sobre self.mejor_ruta (la mejor ruta
        DETERMINISTICA encontrada por el ALNS), calcula el costo esperado
        con la formula cerrada y lo valida contra la version de fuerza
        bruta (enumeracion exhaustiva 2^n). No se llama durante la
        busqueda -- solo cuando ya se decidio cual es la mejor ruta."""
        if self.mejor_ruta is None:
            return

        costo_det, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(self.mejor_ruta)
        if not vehiculos:
            return

        formula, _, _, _ = self.problema.calcular_esperanza_formula_con_vehiculos(
            self.mejor_ruta, vehiculos
        )
        self.mejor_costo_esperado = formula

        n_customers = len([c for c in self.mejor_ruta if c != 0])
        if n_customers > max_n_para_fuerza_bruta:
            self.mejor_costo_esperado_fb = None
            self.mejor_costo_esperado_diff = None
            self.mejor_costo_esperado_fb_skip_reason = (
                f"{n_customers} clientes -> 2^{n_customers} escenarios, "
                f"por encima del limite de {max_n_para_fuerza_bruta}"
            )
            return

        fb = self.problema.calcular_esperanza_fuerza_bruta_con_vehiculos(self.mejor_ruta, vehiculos)
        self.mejor_costo_esperado_fb = fb
        self.mejor_costo_esperado_diff = abs(formula - fb)
        self.mejor_costo_esperado_fb_skip_reason = None

    def ejecutar(self, tiempo_inicio=None, callback_mejora=None, callback_progreso=None, export_callback=None):
        self.callback_mejora = callback_mejora
        self.callback_progreso = callback_progreso
        self.export_callback = export_callback
        self.tiempo_inicio = tiempo_inicio

        ruta_actual = self.solucion_inicial_aleatoria()
        costo_actual = self._evaluar_ruta(ruta_actual)
        self.mejor_ruta = ruta_actual.copy()
        self.mejor_costo = costo_actual
        self.best_iteration = 0
        self.tiempo_mejor = 0.0
        if tiempo_inicio:
            self.tiempo_mejor = time.time() - tiempo_inicio
        self.historial_costos = [costo_actual]

        iter_sin_mejora = 0
        tiempo_ultima_mejora = self.tiempo_mejor

        if callback_mejora:
            costo_det_actual, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(self.mejor_ruta)
            callback_mejora(0, self.mejor_costo, costo_det_actual, self.mejor_ruta, vehiculos, self.tiempo_mejor)

        for it in range(1, self.max_iter+1):
            if self.usar_estanc_iter and iter_sin_mejora >= self.max_estanc_iter:
                break

            if self.usar_estanc_tiempo and tiempo_inicio:
                tiempo_actual = time.time() - tiempo_inicio
                if (tiempo_actual - tiempo_ultima_mejora) >= self.max_estanc_tiempo:
                    break

            d_idx = self.seleccionar_operador(self.destroy_weights)
            r_idx = self.seleccionar_operador(self.repair_weights)

            self.destroy_usage[d_idx] += 1
            self.total_destroy_usage[d_idx] += 1
            self.repair_usage[r_idx] += 1
            self.total_repair_usage[r_idx] += 1

            ruta_parcial, eliminados = self.destroy_ops[d_idx](ruta_actual)
            ruta_nueva = self.repair_ops[r_idx](ruta_parcial, eliminados)
            ruta_nueva = self.busqueda_local_2opt(ruta_nueva)
            costo_nuevo = self._evaluar_ruta(ruta_nueva)

            mejoro = False
            if costo_nuevo < self.mejor_costo:
                self.mejor_ruta = ruta_nueva.copy()
                self.mejor_costo = costo_nuevo
                self.best_iteration = it
                if tiempo_inicio:
                    self.tiempo_mejor = time.time() - tiempo_inicio
                    tiempo_ultima_mejora = self.tiempo_mejor
                puntos = 33
                mejoro = True
                iter_sin_mejora = 0
                if callback_mejora:
                    costo_det_actual, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(self.mejor_ruta)
                    callback_mejora(it, self.mejor_costo, costo_det_actual, self.mejor_ruta, vehiculos, self.tiempo_mejor)
            elif costo_nuevo < costo_actual:
                puntos = 9
            else:
                if self.aceptar(costo_nuevo, costo_actual):
                    puntos = 13
                else:
                    puntos = 0

            self.destroy_scores[d_idx] += puntos
            self.repair_scores[r_idx] += puntos

            if puntos != 0:
                ruta_actual = ruta_nueva
                costo_actual = costo_nuevo

            if not mejoro:
                iter_sin_mejora += 1

            self.temp *= self.cooling
            self.iter_count += 1
            if self.iter_count % self.segment_size == 0:
                self.actualizar_pesos()

            self.historial_costos.append(self.mejor_costo)

            if callback_progreso and (it % 10 == 0 or mejoro):
                callback_progreso(it, self.mejor_costo)

        self._calcular_expected_y_validar()

        if export_callback and self.mejor_ruta:
            _, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(self.mejor_ruta)
            export_callback(self.mejor_ruta, vehiculos, self.mejor_costo, self.best_iteration, "FINAL")

        return self.mejor_ruta, self.mejor_costo


# ------------------------------------------------------------
# 6. API PROGRAMATICA -- para uso desde run_experiments.py
# ------------------------------------------------------------
def run(instance_path: str, seed: int = 42, verbose: bool = False) -> dict:
    """Carga instance_path, corre UNA busqueda ALNS con la semilla dada,
    sin GUI ni graficos. No corre la validacion de fuerza bruta del costo
    esperado (seria demasiado lenta para correr cientos de veces en
    batch) -- esa validacion queda solo para uso manual (ver
    ALNSSolver._calcular_expected_y_validar).

    Hiperparametros del ALNS: los mismos valores por defecto que traia la
    GUI (max_iter=5000, temp_inicial=100, cooling=0.99,
    porc_destruccion=0.3); estancamiento por iteracion (200 sin mejora)
    sigue activo y acota la duracion real de una corrida.

    Devuelve:
        {
            "route": list[int], "vehicles": list[int],
            "cost": float,            # costo deterministico (referencia)
            "expected_cost": float,   # costo esperado PTSP (el objetivo)
            "elapsed_s": float,       # tiempo de la busqueda ALNS
            "best_iteration": int,    # iteracion en la que se hallo la mejor solucion
            "n": int, "k": int,
        }

    Lanza ValueError si la instancia es infactible para la variante 'exato'
    (menos arcos que vehiculos).
    """
    random.seed(seed)
    np.random.seed(seed)

    (dist_matrix, edge_arr, return_arr, n, nv, coords,
     nombre, probs) = cargar_instancia_cars(instance_path)

    problema = CarRenterProblem(n, dist_matrix, edge_arr, return_arr)
    problema.nombre_instancia = nombre
    problema.ruta_instancia = instance_path
    problema.probabilidades = probs
    problema.coordenadas = coords

    solver = ALNSSolver(problema, max_iter=5000, temp_inicial=100,
                         cooling=0.99, porc_destruccion=0.3, verbose=verbose)

    start = time.time()
    mejor_ruta, mejor_costo_esperado = solver.ejecutar()
    elapsed = time.time() - start

    costo_det, vehiculos, _, _ = problema.costo_ruta_con_vehiculos(mejor_ruta)
    if not vehiculos:
        raise ValueError(f"Sin solucion factible (variante 'exato') para {instance_path}")

    return {
        "route": mejor_ruta,
        "vehicles": vehiculos,
        "cost": costo_det,
        "expected_cost": mejor_costo_esperado,
        "elapsed_s": elapsed,
        "best_iteration": solver.best_iteration,
        "n": n,
        "k": nv,
    }


# ------------------------------------------------------------
# 7. EJECUCION PRINCIPAL
# ------------------------------------------------------------
if __name__ == "__main__":
    filename = "instances/euclidean/BrasilRJ14e.car"

    result = run(filename, seed=42, verbose=True)
    route, vehicles = result["route"], result["vehicles"]

    print(f"\n{'=' * 60}")
    print("  RESULTADO (seleccionado por costo esperado PTSP)")
    print(f"{'=' * 60}")
    print(f"  Costo esperado (PTSP):  {result['expected_cost']:.2f}")
    print(f"  Costo deterministico:   {result['cost']:.2f}  (referencia, no es el objetivo)")
    print(f"  Tiempo:                 {result['elapsed_s']:.2f}s")
    print(f"  Mejor iteracion:        {result['best_iteration']}")
    print(f"  Vehiculos utilizados: {sorted(set(vehicles))}  (debe ser {list(range(result['k']))}, variante 'exato')")

    for t in range(len(route) - 1):
        print(f"    {route[t]:3d} -> {route[t + 1]:3d} | Vehiculo {vehicles[t]}")

    # -------------------------------------------------
    # Validaciones manuales (solo al correr este archivo directo, no en batch)
    # -------------------------------------------------
    (dist_matrix, edge_arr, return_arr, n, nv, coords,
     nombre, probs) = cargar_instancia_cars(filename)
    problema = CarRenterProblem(n, dist_matrix, edge_arr, return_arr)
    problema.probabilidades = probs

    formula, t1, t2, t3 = problema.calcular_esperanza_formula_con_vehiculos(route, vehicles)
    n_customers = len([c for c in route if c != 0])
    print(f"\n{'=' * 60}")
    print("  VALIDACION -- formula vs fuerza bruta (costo esperado PTSP)")
    print(f"{'=' * 60}")
    if n_customers <= 20:
        fb = problema.calcular_esperanza_fuerza_bruta_con_vehiculos(route, vehicles)
        diff = abs(formula - fb)
        tag = "MATCH" if diff < 1e-6 else f"MISMATCH (delta={diff:.6f})"
        print(f"  Formula:      {formula:.6f}")
        print(f"  Fuerza bruta: {fb:.6f}  (2^{n_customers} escenarios)")
        print(f"  -> {tag}")
    else:
        print(f"  SALTEADA ({n_customers} clientes -> 2^{n_customers} escenarios)")
    print(f"{'=' * 60}")
