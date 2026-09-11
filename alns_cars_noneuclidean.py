import random
import math
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import matplotlib.colors as mcolors
from collections import defaultdict
import time
from datetime import datetime
import os

# ======================================================
# CONSTANTES DE RUTAS
# ======================================================
RUTA_BASE = r"F:\2026\car"
RUTA_SOLUCIONES = os.path.join(RUTA_BASE, "SOLUCIONES")

if not os.path.exists(RUTA_SOLUCIONES):
    try:
        os.makedirs(RUTA_SOLUCIONES)
        print(f"Carpeta creada: {RUTA_SOLUCIONES}")
    except Exception as e:
        print(f"No se pudo crear la carpeta: {RUTA_SOLUCIONES} - {e}")

# ------------------------------------------------------------
# 1. DEFINICIÓN DEL PROBLEMA CaRS (CON VEHÍCULOS)
# ------------------------------------------------------------
class CarRenterProblem:
    def __init__(self, num_ciudades, dist_matrix, edge_vectors, return_vectors, formato="VECTOR"):
        self.n = num_ciudades
        self.formato = formato
        self.nombre_instancia = "desconocida"
        self.ruta_instancia = ""

        if formato == "VECTOR":
            self.dist = dist_matrix
            self.edge = edge_vectors
            self.return_rate = return_vectors
            self.num_vehiculos = edge_vectors.shape[0] if edge_vectors is not None else 0
            self.edge_matrices = None
        else:  # FULL_MATRIX
            self.dist = None
            self.edge = None
            self.edge_matrices = edge_vectors
            self.return_rate = return_vectors
            self.num_vehiculos = len(edge_vectors) if edge_vectors is not None else 0
            if self.num_vehiculos > 0:
                self.dist = np.array(edge_vectors[0], dtype=float)
            else:
                self.dist = None

        self.costo_arco_cache = {}
        self.probabilidades = None

    def costo_arco_sin_retorno(self, i, j, v):
        if self.formato == "FULL_MATRIX":
            return self.edge_matrices[v][i][j]
        else:
            key = (i, j, v)
            if key not in self.costo_arco_cache:
                term = (self.edge[v][i] * 2 + self.edge[v][j] * 3) / 3.0
                self.costo_arco_cache[key] = term + self.dist[i][j]
            return self.costo_arco_cache[key]

    def costo_retorno(self, ciudad_alquiler, ciudad_devolucion, v):
        # """d^k_ij de la tesis (Silva 2011, p.44 item 3): costo de devolver
        # el vehiculo v, ALQUILADO en ciudad_alquiler, ENTREGADO en
        # ciudad_devolucion. Nulo cuando coinciden (thesis p.44 item 5)."""
        # if ciudad_alquiler == ciudad_devolucion:
        #     return 0.0
        # if self.return_rate is None:
        #     return 0.0
        # if self.formato == "FULL_MATRIX":
        #     return float(self.return_rate[v][ciudad_alquiler][ciudad_devolucion])
        # else:
        #     ri = self.return_rate[v][ciudad_alquiler]
        #     rj = self.return_rate[v][ciudad_devolucion]
        #     dist_ij = self.dist[ciudad_alquiler][ciudad_devolucion]
        #     return float((2 * ri + 3 * rj) / 3.0 + dist_ij)
        return 0.0

    # ------------------------------------------------------------
    # CÁLCULO EXACTO DEL COSTO ESPERADO (FUERZA BRUTA) CON VEHÍCULOS
    # ------------------------------------------------------------
    def calcular_esperanza_fuerza_bruta_con_vehiculos(self, ruta, vehiculos):
        """
        Calcula el valor esperado exacto considerando los vehículos asignados a cada arco.
        vehiculos: lista de longitud L = len(ruta)-1.
        """
        if self.dist is None:
            raise ValueError("No se dispone de matriz de distancias.")
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
            for idx in range(len(subtour)-1):
                a = subtour[idx]
                b = subtour[idx+1]
                if a == 0 and b == 0:
                    continue
                if b == 0:
                    # Termino 2 (ultimo presente -> deposito): vehiculo de
                    # SALIDA del origen a (formula_corregida.html, c_i).
                    pos_a = pos_in_ruta[a]
                    v = vehiculos[pos_a] if pos_a < L else vehiculos[-1]
                else:
                    # Terminos 1 (deposito -> primer presente) y 3 (entre
                    # presentes consecutivos): vehiculo de LLEGADA al
                    # destino b (formula_corregida.html, seccion 9, c_j).
                    pos_b = pos_in_ruta[b]
                    v = vehiculos[pos_b - 1]
                costo += self.costo_arco_sin_retorno(a, b, v)
            total += prob * costo
        return total

    # ------------------------------------------------------------
    # COSTO ESPERADO -- FORMULA CERRADA O(n^2) (T1+T2+T3, sin retorno)
    # Equivalente cerrado de calcular_esperanza_fuerza_bruta_con_vehiculos().
    # Se calcula UNA SOLA VEZ sobre la mejor ruta determinista final; la
    # version de fuerza bruta se usa solo para VALIDARLA (ver ALNSSolver).
    # ------------------------------------------------------------
    def calcular_esperanza_formula_con_vehiculos(self, ruta, vehiculos):
        """
        Formula cerrada del costo esperado PTSP, usando la asignacion de
        vehiculos YA decidida (por costo_ruta_con_vehiculos). Agnostica:
        no optimiza nada, solo mide -- mismo contrato que
        expected_cost_ptsp() en aco_cars_ptsp_exact_noneuclidean.py / aco_cars_ptsp_exact_euclidean.py.

        T1 = sum_j c(v0, r_j, arr_j) * P(r_j) * prod_{k<j} (1 - P(r_k))
        T2 = sum_i c(r_i, v0, dep_i) * P(r_i) * prod_{k>i} (1 - P(r_k))
        T3 = sum_i sum_{j>i} c(r_i, r_j, arr_j) * P(r_i) * P(r_j)
             * prod_{i<v<j} (1 - P(r_v))
        """
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
    # ASIGNACIÓN ÓPTIMA DE VEHÍCULOS (DETERMINISTA)
    # ------------------------------------------------------------
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
        # la misma que aco_cars_ptsp_exact_noneuclidean.py / aco_cars_ptsp_exact_euclidean.py:
        # los num_vehiculos vehiculos deben usarse TODOS al menos una vez.
        # Sin esto el DP podia terminar con 1 solo vehiculo para toda la
        # ruta (degenerando al TSP clasico, thesis Fig. 7) en vez de
        # resolver el mismo problema "exato" que el resto del proyecto.
        #------------------------------------------
        # full_mask = max_mask - 1
        # if L < self.num_vehiculos or dp[L][full_mask] == float('inf'):
        #     return float('inf'), [], set(), []

        # mejor_costo = dp[L][full_mask]
        # mejor_mask = full_mask
        #-----------------------------
        # Sin restriccion de "usar todos los vehiculos": elegir libremente
        # el subconjunto de vehiculos que minimice el costo total.
        mejor_costo = float('inf')
        mejor_mask = None
        for m in range(max_mask):
            if dp[L][m] < mejor_costo:
                mejor_costo = dp[L][m]
                mejor_mask = m

        if mejor_mask is None or mejor_costo == float('inf'):
            return float('inf'), [], set(), []

        
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

    def costo_determinista_con_asignacion(self, ruta, vehiculos):
        """Costo determinista (arcos + retorno) para una asignacion de
        vehiculos DADA por el usuario -- no necesariamente la optima ni
        "exato"/"sem repeticao". Cada tramo contiguo del mismo vehiculo
        paga UN costo de retorno al cerrarse (por cambio de vehiculo o al
        terminar la ruta), igual que verify_cost() en
        aco_cars_ptsp_exact_noneuclidean.py / aco_cars_ptsp_exact_euclidean.py."""
        total = 0.0
        n_arcs = len(ruta) - 1
        block_start_idx = 0

        for t in range(n_arcs):
            i, j, v = ruta[t], ruta[t + 1], vehiculos[t]
            total += self.costo_arco_sin_retorno(i, j, v)

            is_last_arc = t == n_arcs - 1
            switches_next = (not is_last_arc) and vehiculos[t + 1] != v
            if switches_next or is_last_arc:
                origen = ruta[block_start_idx]
                total += self.costo_retorno(origen, j, v)
                block_start_idx = t + 1

        return total

    def costo_variable_ruta(self, segmento):
        if len(segmento) != 2:
            raise ValueError("costo_variable_ruta solo soporta segmentos de 2 ciudades")
        i, j = segmento
        if self.formato == "FULL_MATRIX":
            return min(self.edge_matrices[v][i][j] for v in range(self.num_vehiculos))
        else:
            return self.costo_arco_sin_retorno(i, j, 0)

    def exportar_solucion(self, ruta, vehiculos_por_arco, costo_esperado, costo_vehiculos, iteracion, hilo_id,
                           motivo="final", costo_esperado_fb=None, diff_esperado=None):
        nombre_base = os.path.splitext(os.path.basename(self.nombre_instancia))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sufijo = {"pausa": "PAUSA", "stop": "STOP", "final": "FINAL"}.get(motivo, "FINAL")
        nombre_archivo = f"{nombre_base}_mejor_ruta_{sufijo}_{timestamp}.carsol"
        archivo_salida = os.path.join(RUTA_SOLUCIONES, nombre_archivo)

        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write("#" + "="*78 + "\n")
            f.write("# ARCHIVO DE SOLUCIÓN CaRS (busqueda ALNS: costo DETERMINISTICO;\n")
            f.write("# costo esperado PTSP calculado una sola vez sobre la ruta final)\n")
            f.write("#" + "="*78 + "\n")
            f.write(f"INSTANCIA: {self.nombre_instancia}\n")
            f.write(f"FECHA: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"HORA: {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"COSTO_DETERMINISTICO: {costo_vehiculos:.6f}\n")
            if costo_esperado is not None:
                f.write(f"COSTO_ESPERADO_FORMULA: {costo_esperado:.6f}\n")
            if costo_esperado_fb is not None:
                f.write(f"COSTO_ESPERADO_FUERZA_BRUTA: {costo_esperado_fb:.6f}\n")
                f.write(f"DIFERENCIA_FORMULA_VS_FB: {diff_esperado:.10f}\n")
            f.write(f"ITERACION: {iteracion}\n")
            f.write(f"HILO: {hilo_id}\n")
            f.write(f"MOTIVO: {sufijo}\n")
            f.write("#" + "="*78 + "\n")
            f.write("RUTA:\n")
            f.write(" ".join(map(str, ruta)) + "\n")
            f.write("#" + "="*78 + "\n")
            f.write("VEHICULOS_POR_ARCO:\n")
            f.write(" ".join(str(v) for v in vehiculos_por_arco) + "\n")
            f.write("#" + "="*78 + "\n")
            tipos_usados = sorted(set(vehiculos_por_arco))
            f.write("RESUMEN_VEHICULOS:\n")
            f.write(" ".join(str(v) for v in tipos_usados) + "\n")
            f.write("#" + "="*78 + "\n")
            f.write("FIN_ARCHIVO\n")
        return archivo_salida


# ------------------------------------------------------------
# 2. CARGA DE INSTANCIA (sin cambios)
# ------------------------------------------------------------
def cargar_instancia_cars(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        lineas = [line.rstrip('\n') for line in f]

    n = None
    n_vehiculos = None
    coordenadas = []
    edge_vectors = None
    return_vectors = None
    nombre_instancia = os.path.basename(ruta_archivo)
    weight_format = "VECTOR"
    formato = "VECTOR"
    probabilidades = None

    i = 0
    total = len(lineas)
    estado = None
    weight_type = "EUC_2D"

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
        elif linea.startswith('EDGE_WEIGHT_TYPE'):
            weight_type = linea.split(':')[1].strip()
        elif linea.startswith('EDGE_WEIGHT_FORMAT'):
            weight_format = linea.split(':')[1].strip()
            formato = weight_format
        elif linea.startswith('NODE_COORD_SECTION'):
            estado = 'COORDS'
            continue
        elif linea.startswith('EDGE_WEIGHT_SECTION'):
            estado = 'EDGE'
            if weight_format == "VECTOR":
                edge_vectors = [None] * n_vehiculos
            else:
                edge_vectors = []
            continue
        elif linea.startswith('RETURN_RATE_SECTION'):
            estado = 'RETURN'
            if weight_format == "VECTOR":
                return_vectors = [None] * n_vehiculos
            else:
                return_vectors = []
            continue
        elif linea.startswith('PROBABILITY_SECTION'):
            estado = 'PROB'
            probabilidades_raw = []
            while i < total:
                linea_actual = lineas[i].strip()
                if not linea_actual:
                    i += 1
                    continue
                if linea_actual.startswith('EOF') or linea_actual.startswith('NAME') or linea_actual.startswith('DIMENSION'):
                    break
                partes = linea_actual.split()
                if not partes:
                    i += 1
                    continue
                try:
                    fila = [float(x) for x in partes]
                    probabilidades_raw.extend(fila)
                except ValueError:
                    break
                i += 1
            if len(probabilidades_raw) == n * n:
                matriz_prob = np.array(probabilidades_raw).reshape(n, n)
                probs = list(matriz_prob[0, 1:])
                probabilidades = [1.0] + probs
                if len(probabilidades) != n:
                    raise ValueError(f"Se esperaban {n} probabilidades, se obtuvieron {len(probabilidades)}")
            elif len(probabilidades_raw) == n:
                probabilidades = probabilidades_raw
            else:
                raise ValueError(f"Se esperaban {n*n} o {n} números en la sección de probabilidades, se obtuvieron {len(probabilidades_raw)}")
            estado = None
            continue

        if estado == 'COORDS':
            partes = linea.split()
            if len(partes) >= 3:
                x = float(partes[1])
                y = float(partes[2])
                coordenadas.append((x, y))
            continue

        if estado in ('EDGE', 'RETURN'):
            if not linea.isdigit():
                continue
            idx = int(linea)
            if weight_format == "VECTOR":
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
            else:
                matriz = []
                for _ in range(n):
                    while i < total and lineas[i].strip() == '':
                        i += 1
                    if i >= total:
                        break
                    fila = list(map(int, lineas[i].split()))
                    if len(fila) != n:
                        raise ValueError(f"Fila de matriz incompleta: se esperaban {n} elementos")
                    matriz.append(fila)
                    i += 1
                if estado == 'EDGE':
                    edge_vectors.append(np.array(matriz, dtype=float))
                else:
                    return_vectors.append(np.array(matriz, dtype=float))
            continue

    if n is None or n_vehiculos is None:
        raise ValueError("No se encontraron DIMENSION o CARS_NUMBER")

    if weight_format == "VECTOR":
        if any(v is None for v in edge_vectors):
            raise ValueError("EDGE_WEIGHT_SECTION incompleta")
        if any(v is None for v in return_vectors):
            raise ValueError("RETURN_RATE_SECTION incompleta")
        edge_arr = np.array(edge_vectors, dtype=float)
        return_arr = np.array(return_vectors, dtype=float)
        if len(coordenadas) == n:
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                xi, yi = coordenadas[i]
                for j in range(n):
                    if i != j:
                        xj, yj = coordenadas[j]
                        dx = xi - xj
                        dy = yi - yj
                        dist_matrix[i][j] = math.sqrt(dx*dx + dy*dy)
        else:
            dist_matrix = np.zeros((n, n))
        edge_matrices = None
    else:
        if not edge_vectors:
            raise ValueError("No se encontraron matrices en EDGE_WEIGHT_SECTION")
        edge_arr = None
        return_arr = None
        dist_matrix = edge_vectors[0]
        edge_matrices = edge_vectors
        if return_vectors:
            return_arr = np.array(return_vectors)

    if probabilidades is None:
        probabilidades = [1.0] * n
        print("Advertencia: No se encontró sección de probabilidades. Se asume 1.0 para todas las ciudades.")

    return dist_matrix, edge_arr, edge_matrices, return_arr, n, n_vehiculos, coordenadas, nombre_instancia, formato, probabilidades


# ------------------------------------------------------------
# 3. FUNCIONES DE GRAFICADO (sin cambios)
# ------------------------------------------------------------
def graficar_ciudades(coordenadas, titulo):
    if not coordenadas or len(coordenadas) < 2:
        print("No hay coordenadas para graficar la distribución.")
        return
    coords = np.array(coordenadas)
    plt.figure(figsize=(8,6))
    plt.scatter(coords[:,0], coords[:,1], c='blue', s=50, label='Ciudades')
    plt.scatter(coords[0,0], coords[0,1], c='red', s=100, marker='s', label='Depósito')
    for i, (x,y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=9, ha='center', va='bottom')
    plt.title(titulo)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def graficar_ruta_con_vehiculos(coordenadas, ruta, vehiculos_por_arco, tipos_usados, costo_total, problema):
    if not coordenadas or len(coordenadas) < 2:
        n = problema.n
        coords_list = [(math.cos(2*math.pi*i/n)*10, math.sin(2*math.pi*i/n)*10) for i in range(n)]
        coords = np.array(coords_list)
        print("No se tienen coordenadas reales; se usará un layout circular para la visualización.")
    else:
        coords = np.array(coordenadas)

    colores = list(mcolors.TABLEAU_COLORS.values())
    color_por_vehiculo = {t: colores[t % len(colores)] for t in tipos_usados}

    retornos = []
    if vehiculos_por_arco and len(vehiculos_por_arco) > 0:
        v_anterior = vehiculos_por_arco[0]
        for idx in range(1, len(vehiculos_por_arco)):
            v_actual = vehiculos_por_arco[idx]
            if v_actual != v_anterior:
                ciudad_devolucion = ruta[idx]
                if ciudad_devolucion != 0:
                    ciudad_alquiler = ruta[idx-1]
                    retornos.append((ciudad_alquiler, ciudad_devolucion, v_anterior))
            v_anterior = v_actual
        ultimo_v = vehiculos_por_arco[-1]
        if ultimo_v is not None:
            for idx, v in enumerate(vehiculos_por_arco):
                if v == ultimo_v:
                    ciudad_alquiler = ruta[idx]
                    if ciudad_alquiler != 0:
                        retornos.append((ciudad_alquiler, 0, ultimo_v))
                    break

    from collections import defaultdict
    retornos_por_alquiler = defaultdict(list)
    for ciudad_alquiler, ciudad_devolucion, v in retornos:
        retornos_por_alquiler[ciudad_alquiler].append((ciudad_devolucion, v))

    costo_total_arcos = 0.0
    for idx in range(len(ruta)-1):
        i, j = ruta[idx], ruta[idx+1]
        v = vehiculos_por_arco[idx]
        if v is None:
            continue
        costo_arco = problema.costo_arco_sin_retorno(i, j, v)
        costo_total_arcos += costo_arco

    costo_total_retornos = 0.0
    for alq, lista in retornos_por_alquiler.items():
        for dev, v in lista:
            costo_total_retornos += problema.costo_retorno(alq, dev, v)

    plt.figure(figsize=(16, 14))
    plt.scatter(coords[:,0], coords[:,1], c='lightgray', s=80, edgecolors='black', zorder=2)
    plt.scatter(coords[0,0], coords[0,1], c='gold', s=200, marker='s', edgecolors='black', label='Depósito (0)', zorder=3)

    for idx in range(len(ruta)-1):
        i, j = ruta[idx], ruta[idx+1]
        v = vehiculos_por_arco[idx]
        if v is None:
            continue
        color = color_por_vehiculo.get(v, 'black')
        plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]],
                 color=color, linewidth=3, alpha=0.8, zorder=1)
        costo_arco = problema.costo_arco_sin_retorno(i, j, v)
        if problema.formato == "VECTOR":
            dist_ij = problema.dist[i][j]
            edge_i = problema.edge[v][i]
            edge_j = problema.edge[v][j]
            term_ew = (edge_i * 2 + edge_j * 3) / 3.0
            texto = f"{idx+1}\nd={dist_ij:.2f}\new={term_ew:.2f}\narc={costo_arco:.2f}"
        else:
            texto = f"{idx+1}\narc={costo_arco:.2f}"
        mid_x = (coords[i,0] + coords[j,0]) / 2
        mid_y = (coords[i,1] + coords[j,1]) / 2
        plt.text(mid_x, mid_y, texto, fontsize=7, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor=color),
                 color='black', zorder=5)

    deposito = coords[0]
    offset_scale = 2.0
    for ciudad_alquiler, lista_ret in retornos_por_alquiler.items():
        if not lista_ret:
            continue
        x0, y0 = coords[ciudad_alquiler, 0], coords[ciudad_alquiler, 1]
        n_ret = len(lista_ret)
        offsets = np.linspace(-offset_scale, offset_scale, n_ret) if n_ret > 1 else [0.0]
        for (ciudad_devolucion, v), off in zip(lista_ret, offsets):
            xd, yd = deposito[0], deposito[1]
            dx = xd - x0
            dy = yd - y0
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            ux = dx / length
            uy = dy / length
            px = -uy
            py = ux
            x_start = x0 + px * off
            y_start = y0 + py * off
            color = color_por_vehiculo.get(v, 'red')
            plt.plot([x_start, xd], [y_start, yd],
                     color=color, linewidth=3, linestyle='--', alpha=0.9, zorder=1)
            plt.annotate('', xy=(xd, yd), xytext=(x_start, y_start),
                         arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.9, linestyle='--'))
            circle = plt.Circle((x_start, y_start), 0.5, color=color, fill=False, linewidth=3, linestyle='--', zorder=4)
            plt.gca().add_patch(circle)
            costo_ret = problema.costo_retorno(ciudad_alquiler, ciudad_devolucion, v)
            if problema.formato == "VECTOR":
                termino_rr = (problema.return_rate[v][ciudad_alquiler] * 2 + problema.return_rate[v][0] * 3) / 3.0
                distancia = problema.dist[ciudad_alquiler][ciudad_devolucion]
            else:
                termino_rr = problema.return_rate[v][ciudad_alquiler][ciudad_devolucion]
                distancia = problema.edge_matrices[v][ciudad_alquiler][ciudad_devolucion]
            mid_x = (x_start + xd) / 2
            mid_y = (y_start + yd) / 2
            texto_ret = f"RET v{v}: {costo_ret:.2f}\n(d={distancia:.2f} + rr={termino_rr:.2f})"
            plt.text(mid_x, mid_y, texto_ret, fontsize=8, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.95),
                     color=color, zorder=6)

    for idx in range(len(ruta)-1):
        i, j = ruta[idx], ruta[idx+1]
        dx = coords[j,0] - coords[i,0]
        dy = coords[j,1] - coords[i,1]
        if dx != 0 or dy != 0:
            plt.annotate('', xy=(coords[j,0], coords[j,1]), xytext=(coords[i,0], coords[i,1]),
                         arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.6))

    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=11, ha='center', va='bottom', fontweight='bold', zorder=4)

    legend_elements = [plt.Line2D([0], [0], color=color_por_vehiculo[v], lw=3, label=f'Vehículo {v}')
                       for v in sorted(tipos_usados)]
    if retornos:
        for v in sorted(set(v for _, _, v in retornos)):
            legend_elements.append(plt.Line2D([0], [0], color=color_por_vehiculo[v], lw=2, linestyle='--',
                                              label=f'Retorno vehículo {v}'))
    plt.legend(handles=legend_elements, title='Tipo de vehículo', loc='best')

    titulo = f"SOLUCIÓN DEL PROBLEMA CaRS (costo con vehículos: {costo_total:.2f})\n"
    titulo += f"{'='*60}\n"
    titulo += f"Suma de arcos: {costo_total_arcos:.2f}  |  Suma de retornos: {costo_total_retornos:.2f}\n"
    if retornos:
        titulo += "Retornos:\n" + "\n".join(f"  • Vehículo {v} → desde ciudad {alq} hasta ciudad {dev}: {problema.costo_retorno(alq, dev, v):.2f}"
                                            for alq, dev, v in retornos)
    else:
        titulo += "No hay costos de retorno."
    plt.title(titulo, fontsize=10, loc='left', fontfamily='monospace')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_ruta_esperanza(coordenadas, ruta, probabilidades, valor_esperado, problema):
    if not coordenadas or len(coordenadas) < 2:
        n = problema.n
        coords_list = [(math.cos(2*math.pi*i/n)*10, math.sin(2*math.pi*i/n)*10) for i in range(n)]
        coords = np.array(coords_list)
    else:
        coords = np.array(coordenadas)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_title(f"Valor esperado E[S] = {valor_esperado:.6f}", fontsize=14)

    for i, (x, y) in enumerate(coords):
        color = 'green' if probabilidades[i] > 0.5 else 'gray'
        marker = 'o' if probabilidades[i] > 0.5 else 's'
        ax.plot(x, y, marker=marker, color=color, markersize=10)
        ax.text(x+0.3, y+0.3, f"{i}\np={probabilidades[i]:.2f}", fontsize=9)

    ax.plot(coords[0,0], coords[0,1], 'ks', markersize=14, label='Depósito (0)')
    ax.text(coords[0,0]-0.5, coords[0,1]-0.5, 'v0', fontsize=12, fontweight='bold')

    orden = [c for c in ruta if c != 0]
    puntos = [coords[i] for i in ruta]
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    ax.plot(xs, ys, 'k--', alpha=0.2, linewidth=1, label='Ruta fija (todos)')

    if not orden:
        ax.text(0.5, 0.5, "No hay ciudades en la ruta", transform=ax.transAxes, ha='center')
        plt.show()
        return

    i0 = orden[0]
    ax.annotate("", xy=coords[i0], xytext=coords[0],
                arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8))
    ax.text((coords[0,0]+coords[i0,0])/2, (coords[0,1]+coords[i0,1])/2,
            f"Salida\nd={np.linalg.norm(coords[0]-coords[i0]):.2f}",
            color='red', fontsize=10, ha='center', va='bottom')

    for idx in range(len(orden)-1):
        ci = orden[idx]
        cj = orden[idx+1]
        ax.annotate("", xy=coords[cj], xytext=coords[ci],
                    arrowprops=dict(arrowstyle='->', color='green', lw=3, alpha=0.8))
        ax.text((coords[ci,0]+coords[cj,0])/2, (coords[ci,1]+coords[cj,1])/2,
                f"{idx+1}\nd={np.linalg.norm(coords[ci]-coords[cj]):.2f}",
                color='green', fontsize=10, ha='center', va='bottom')

    ultimo = orden[-1]
    ax.annotate("", xy=coords[0], xytext=coords[ultimo],
                arrowprops=dict(arrowstyle='->', color='blue', lw=3, alpha=0.8))
    ax.text((coords[ultimo,0]+coords[0,0])/2, (coords[ultimo,1]+coords[0,1])/2,
            f"Regreso\nd={np.linalg.norm(coords[ultimo]-coords[0]):.2f}",
            color='blue', fontsize=10, ha='center', va='bottom')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Salida (v0 → primer presente)'),
        Patch(facecolor='green', edgecolor='green', label='Entre presentes consecutivos'),
        Patch(facecolor='blue', edgecolor='blue', label='Regreso (último presente → v0)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4. ALGORITMO ALNS (OPTIMIZA ESPERANZA CON VEHÍCULOS)
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
        self.pausa_event = threading.Event()
        self.stop_event = threading.Event()
        self.pausado = False

        self.usar_estanc_iter = usar_estanc_iter
        self.max_estanc_iter = max_estanc_iter
        self.usar_estanc_tiempo = usar_estanc_tiempo
        self.max_estanc_tiempo = max_estanc_tiempo

    def _evaluar_ruta(self, ruta):
        """Costo DETERMINISTICO (arcos + retorno correcto, thesis d^k_ij)
        de la ruta, con la asignacion optima de vehiculos via DP -- este
        es el objetivo que busca el ALNS. El costo esperado (PTSP) se
        calcula UNA SOLA VEZ, al final, sobre la mejor ruta encontrada
        (ver ALNSSolver._calcular_expected_y_validar) -- no en cada
        evaluacion de la busqueda, para no pagar el costo de la formula
        (o peor, de la fuerza bruta) miles de veces por corrida."""
        if len(ruta) < 2:
            return 0.0
        costo, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(ruta)
        if not vehiculos:
            return float('inf')
        #return costo
        costo_esp, _, _, _ = self.problema.calcular_esperanza_formula_con_vehiculos(ruta, vehiculos)
        return costo_esp                      # ← devuelve ESPERADO

    # Operadores de destrucción y reparación (sin cambios, igual que antes)
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
        distancias = [(i, self.problema.costo_variable_ruta([seed, c])) for i, c in enumerate(clientes) if i != seed_idx]
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
        max_val = np.max([self.problema.costo_variable_ruta([0, c]) for c in range(self.problema.n)])
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
                noise = random.uniform(-max_val*mu, max_val*mu)
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

    def pausar(self):
        self.pausa_event.set()
        self.pausado = True

    def reanudar(self):
        self.pausa_event.clear()
        self.pausado = False

    def detener(self):
        self.stop_event.set()

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
            _, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(self.mejor_ruta)
            callback_mejora(0, self.mejor_costo, self.mejor_ruta, vehiculos, self.tiempo_mejor)

        for it in range(1, self.max_iter+1):
            if self.pausa_event.is_set():
                if export_callback and self.mejor_ruta:
                    _, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(self.mejor_ruta)
                    export_callback(self.mejor_ruta, vehiculos, self.mejor_costo, self.best_iteration, "PAUSA")
                while self.pausa_event.is_set():
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.1)

            if self.stop_event.is_set():
                self._calcular_expected_y_validar()
                if export_callback and self.mejor_ruta:
                    _, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(self.mejor_ruta)
                    export_callback(self.mejor_ruta, vehiculos, self.mejor_costo, self.best_iteration, "STOP")
                return self.mejor_ruta, self.mejor_costo

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
                    _, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(self.mejor_ruta)
                    callback_mejora(it, self.mejor_costo, self.mejor_ruta, vehiculos, self.tiempo_mejor)
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
# 5. INTERFAZ GRÁFICA (con verificación de consistencia)
# ------------------------------------------------------------
class CarsGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CaRS Solver - Optimización con vehículos")
        self.root.geometry("1000x950")
        self.problem = None
        self.solver = None
        self.coordenadas = None
        self.ultimo_archivo_exportado = None
        self.tiempo_inicio = 0
        self.var_mostrar_grafico = tk.BooleanVar(value=False)
        self.var_usar_estanc_iter = tk.BooleanVar(value=True)
        self.var_usar_estanc_tiempo = tk.BooleanVar(value=True)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frame_solver = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_solver, text="Solver")

        self.frame_evaluador = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_evaluador, text="Evaluador de Ruta")

        self._create_solver_widgets()
        self._create_evaluador_widgets()

    def _create_solver_widgets(self):
        frame_top = ttk.LabelFrame(self.frame_solver, text="1. Cargar instancia CaRS", padding=5)
        frame_top.pack(fill=tk.X, padx=10, pady=5)
        row1 = ttk.Frame(frame_top)
        row1.pack(fill=tk.X)
        self.btn_select = ttk.Button(row1, text="Seleccionar archivo", command=self.cargar_archivo)
        self.btn_select.pack(side=tk.LEFT, padx=5)
        self.lbl_file = ttk.Label(row1, text="Ningún archivo cargado")
        self.lbl_file.pack(side=tk.LEFT, padx=10)
        row2 = ttk.Frame(frame_top)
        row2.pack(fill=tk.X, pady=2)
        self.chk_mostrar_grafico = ttk.Checkbutton(row2, text="Mostrar distribución de ciudades al cargar",
                                                   variable=self.var_mostrar_grafico)
        self.chk_mostrar_grafico.pack(side=tk.LEFT, padx=5)

        frame_params = ttk.LabelFrame(self.frame_solver, text="2. Parámetros ALNS", padding=5)
        frame_params.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame_params, text="Iteraciones:").grid(row=0, column=0, sticky=tk.W)
        self.entry_iter = ttk.Entry(frame_params, width=10)
        self.entry_iter.insert(0, "5000")
        self.entry_iter.grid(row=0, column=1, padx=5)
        ttk.Label(frame_params, text="Temperatura inicial:").grid(row=0, column=2, sticky=tk.W)
        self.entry_temp = ttk.Entry(frame_params, width=10)
        self.entry_temp.insert(0, "100")
        self.entry_temp.grid(row=0, column=3, padx=5)
        ttk.Label(frame_params, text="Enfriamiento:").grid(row=1, column=0, sticky=tk.W)
        self.entry_cool = ttk.Entry(frame_params, width=10)
        self.entry_cool.insert(0, "0.99")
        self.entry_cool.grid(row=1, column=1, padx=5)
        ttk.Label(frame_params, text="% Destrucción:").grid(row=1, column=2, sticky=tk.W)
        self.entry_destr = ttk.Entry(frame_params, width=10)
        self.entry_destr.insert(0, "0.3")
        self.entry_destr.grid(row=1, column=3, padx=5)

        frame_estanc = ttk.LabelFrame(self.frame_solver, text="Criterios de estancamiento (opcionales)", padding=5)
        frame_estanc.pack(fill=tk.X, padx=10, pady=5)
        f1 = ttk.Frame(frame_estanc)
        f1.pack(fill=tk.X, pady=2)
        self.chk_estanc_iter = ttk.Checkbutton(f1, text="Estancamiento por iteraciones",
                                               variable=self.var_usar_estanc_iter)
        self.chk_estanc_iter.pack(side=tk.LEFT, padx=5)
        ttk.Label(f1, text="Máximo sin mejora (iter):").pack(side=tk.LEFT, padx=5)
        self.entry_estanc_iter = ttk.Entry(f1, width=8)
        self.entry_estanc_iter.insert(0, "200")
        self.entry_estanc_iter.pack(side=tk.LEFT, padx=5)
        f2 = ttk.Frame(frame_estanc)
        f2.pack(fill=tk.X, pady=2)
        self.chk_estanc_tiempo = ttk.Checkbutton(f2, text="Estancamiento por tiempo",
                                                 variable=self.var_usar_estanc_tiempo)
        self.chk_estanc_tiempo.pack(side=tk.LEFT, padx=5)
        ttk.Label(f2, text="Máximo sin mejora (s):").pack(side=tk.LEFT, padx=5)
        self.entry_estanc_tiempo = ttk.Entry(f2, width=8)
        self.entry_estanc_tiempo.insert(0, "60")
        self.entry_estanc_tiempo.pack(side=tk.LEFT, padx=5)

        frame_controls = ttk.LabelFrame(self.frame_solver, text="3. Control", padding=5)
        frame_controls.pack(fill=tk.X, padx=10, pady=5)
        self.btn_run = ttk.Button(frame_controls, text="▶ Ejecutar", command=self.ejecutar)
        self.btn_run.pack(side=tk.LEFT, padx=5)
        self.btn_pause = ttk.Button(frame_controls, text="⏸ Pausa", command=self.pausar, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        self.btn_resume = ttk.Button(frame_controls, text="▶ Reanudar", command=self.reanudar, state=tk.DISABLED)
        self.btn_resume.pack(side=tk.LEFT, padx=5)
        self.btn_stop = ttk.Button(frame_controls, text="⏹ Stop", command=self.detener, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        self.btn_export = ttk.Button(frame_controls, text="💾 Exportar ahora", command=self.exportar_ahora, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=5)

        frame_info = ttk.LabelFrame(self.frame_solver, text="4. Mejor solución encontrada", padding=5)
        frame_info.pack(fill=tk.X, padx=10, pady=5)
        self.lbl_mejor_costo = ttk.Label(frame_info, text="Costo esperado (ALNS): --", font=("Arial", 10, "bold"))
        self.lbl_mejor_costo.pack(anchor=tk.W)
        self.lbl_mejor_iter = ttk.Label(frame_info, text="Iteración mejor solución: --")
        self.lbl_mejor_iter.pack(anchor=tk.W)
        self.lbl_mejor_tiempo = ttk.Label(frame_info, text="Tiempo mejor solución: --")
        self.lbl_mejor_tiempo.pack(anchor=tk.W)
        self.lbl_mejor_ruta = ttk.Label(frame_info, text="Mejor ruta: --", wraplength=800, justify=tk.LEFT)
        self.lbl_mejor_ruta.pack(anchor=tk.W)
        self.lbl_vehiculos_arco = ttk.Label(frame_info, text="Vehículos por arco (óptimos): --", wraplength=800, justify=tk.LEFT)
        self.lbl_vehiculos_arco.pack(anchor=tk.W)
        self.lbl_tipos_usados = ttk.Label(frame_info, text="Tipos usados: --")
        self.lbl_tipos_usados.pack(anchor=tk.W)
        self.lbl_tiempo_total = ttk.Label(frame_info, text="Tiempo total de ejecución: --")
        self.lbl_tiempo_total.pack(anchor=tk.W)
        self.lbl_estado = ttk.Label(frame_info, text="Estado: Esperando")
        self.lbl_estado.pack(anchor=tk.W)

        # Nueva etiqueta para verificación
        self.lbl_verificacion = ttk.Label(frame_info, text="Verificación: --", foreground="blue")
        self.lbl_verificacion.pack(anchor=tk.W)

        frame_result = ttk.LabelFrame(self.frame_solver, text="5. Log de mejoras", padding=5)
        frame_result.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.txt_result = tk.Text(frame_result, wrap=tk.WORD, height=15)
        scroll = ttk.Scrollbar(frame_result, orient=tk.VERTICAL, command=self.txt_result.yview)
        self.txt_result.configure(yscrollcommand=scroll.set)
        self.txt_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.lbl_status = ttk.Label(self.frame_solver, text="Estado: Esperando acción")
        self.lbl_status.pack(pady=5)

    def _create_evaluador_widgets(self):
        # (igual que antes, sin cambios)
        frame_carga = ttk.LabelFrame(self.frame_evaluador, text="Cargar instancia", padding=5)
        frame_carga.pack(fill=tk.X, padx=10, pady=5)

        self.btn_cargar_eval = ttk.Button(frame_carga, text="Seleccionar archivo", command=self.cargar_archivo_eval)
        self.btn_cargar_eval.pack(side=tk.LEFT, padx=5)
        self.lbl_file_eval = ttk.Label(frame_carga, text="Ningún archivo cargado")
        self.lbl_file_eval.pack(side=tk.LEFT, padx=10)

        frame_calculo = ttk.LabelFrame(self.frame_evaluador, text="Ingrese ruta y vehículos", padding=10)
        frame_calculo.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame_calculo, text="Ruta (números separados por espacios, ej: 0 5 8 6 ... 0):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_ruta = ttk.Entry(frame_calculo, width=60)
        self.entry_ruta.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        ttk.Label(frame_calculo, text="Vehículos por arco (opcional, ej: 1 1 1 0 0 ...):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_vehiculos = ttk.Entry(frame_calculo, width=60)
        self.entry_vehiculos.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

        self.btn_calcular = ttk.Button(frame_calculo, text="Calcular costos", command=self.evaluar_ruta_manual, state=tk.DISABLED)
        self.btn_calcular.grid(row=2, column=1, pady=10, sticky=tk.W)

        # Área de resultados
        self.lbl_ruta_evaluada = ttk.Label(frame_calculo, text="Ruta evaluada: --", font=("Arial", 10))
        self.lbl_ruta_evaluada.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.lbl_esperanza_opt = ttk.Label(frame_calculo, text="Costo esperado (óptimo) con vehículos: --", font=("Arial", 10, "bold"))
        self.lbl_esperanza_opt.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.lbl_asignacion_opt = ttk.Label(frame_calculo, text="Asignación óptima de vehículos: --", font=("Arial", 10))
        self.lbl_asignacion_opt.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.lbl_costo_det_opt = ttk.Label(frame_calculo, text="Costo determinista (óptimo): --", font=("Arial", 10))
        self.lbl_costo_det_opt.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.lbl_esperanza_ing = ttk.Label(frame_calculo, text="Costo esperado (ingresado) con vehículos: --", font=("Arial", 10))
        self.lbl_esperanza_ing.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.lbl_costo_det_ing = ttk.Label(frame_calculo, text="Costo determinista (ingresado): --", font=("Arial", 10))
        self.lbl_costo_det_ing.grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.lbl_diff = ttk.Label(frame_calculo, text="Diferencia (ingresado - óptimo): --", font=("Arial", 10))
        self.lbl_diff.grid(row=9, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.lbl_tiempo_eval = ttk.Label(frame_calculo, text="Tiempo de cálculo: --")
        self.lbl_tiempo_eval.grid(row=10, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.lbl_estado_eval = ttk.Label(frame_calculo, text="Cargue una instancia para habilitar el cálculo.", foreground="gray")
        self.lbl_estado_eval.grid(row=11, column=0, columnspan=3, sticky=tk.W, pady=10)

    # ------------------------------------------------------------
    # Métodos de carga y evaluación (sin cambios)
    # ------------------------------------------------------------
    def cargar_instancia_desde_archivo(self, archivo):
        try:
            dist_matrix, edge_arr, edge_matrices, return_arr, n, nv, coords, nombre, formato, probs = cargar_instancia_cars(archivo)
            self.coordenadas = coords
            self.problem = CarRenterProblem(n, dist_matrix, edge_matrices, return_arr, formato=formato)
            self.problem.nombre_instancia = nombre
            self.problem.ruta_instancia = archivo
            self.problem.probabilidades = probs

            nombre_archivo = os.path.basename(archivo)
            self.lbl_file.config(text=nombre_archivo)
            self.lbl_file_eval.config(text=nombre_archivo)

            self.btn_run.config(state=tk.NORMAL)
            self.btn_calcular.config(state=tk.NORMAL)
            self.lbl_estado_eval.config(text="Instancia cargada. Ingrese ruta y vehículos (opcional).", foreground="green")

            if self.var_mostrar_grafico.get() and coords:
                graficar_ciudades(coords, f"Distribución - {nombre_archivo}")
            elif self.var_mostrar_grafico.get() and not coords:
                messagebox.showinfo("Información", "El archivo no contiene coordenadas; no se puede mostrar la distribución.")

            msg = f"Instancia cargada: {n} ciudades, {nv} vehículos\nFormato: {formato}\nProbabilidades leídas: {len(probs)} valores"
            messagebox.showinfo("Éxito", msg)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")

    def cargar_archivo(self):
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de instancia CaRS",
            initialdir="instances/noneuclidean",
            filetypes=[("Archivos CaRS", "*.car *.txt"), ("Todos los archivos", "*.*")]
        )
        if archivo:
            self.cargar_instancia_desde_archivo(archivo)

    def cargar_archivo_eval(self):
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de instancia CaRS",
            initialdir="instances/noneuclidean",
            filetypes=[("Archivos CaRS", "*.car *.txt"), ("Todos los archivos", "*.*")]
        )
        if archivo:
            self.cargar_instancia_desde_archivo(archivo)

    def evaluar_ruta_manual(self):
        if self.problem is None:
            messagebox.showwarning("Advertencia", "Primero cargue una instancia.")
            return

        texto_ruta = self.entry_ruta.get().strip()
        if not texto_ruta:
            messagebox.showwarning("Advertencia", "Ingrese una ruta.")
            return
        try:
            ruta = list(map(int, texto_ruta.split()))
        except ValueError:
            messagebox.showerror("Error", "La ruta debe contener solo números enteros separados por espacios.")
            return

        if ruta[0] != 0 or ruta[-1] != 0:
            messagebox.showerror("Error", "La ruta debe comenzar y terminar en 0 (depósito).")
            return
        n = self.problem.n
        for c in ruta:
            if c < 0 or c >= n:
                messagebox.showerror("Error", f"Ciudad {c} fuera de rango (0..{n-1}).")
                return
        sin_deposito = ruta[1:-1]
        if len(set(sin_deposito)) != len(sin_deposito):
            messagebox.showerror("Error", "Hay ciudades duplicadas en la ruta (sin contar el depósito).")
            return

        texto_veh = self.entry_vehiculos.get().strip()
        vehiculos_ingresados = None
        if texto_veh:
            try:
                vehiculos_ingresados = list(map(int, texto_veh.split()))
            except ValueError:
                messagebox.showerror("Error", "Los vehículos deben ser números enteros separados por espacios.")
                return
            L = len(ruta) - 1
            if len(vehiculos_ingresados) != L:
                messagebox.showerror("Error", f"Debe ingresar exactamente {L} vehículos (uno por arco).")
                return
            for v in vehiculos_ingresados:
                if v < 0 or v >= self.problem.num_vehiculos:
                    messagebox.showerror("Error", f"Vehículo {v} fuera de rango (0..{self.problem.num_vehiculos-1}).")
                    return

        start_time = time.time()
        try:
            costo_det_opt, asignacion_opt, tipos_opt, _ = self.problem.costo_ruta_con_vehiculos(ruta)

            # Costo esperado: FORMULA cerrada + FUERZA BRUTA como validacion
            # cruzada (no solo fuerza bruta, y la fuerza bruta se salta con
            # aviso -- no revienta el calculo entero -- si hay demasiados
            # clientes para enumerar 2^n escenarios).
            formula_opt, _, _, _ = self.problem.calcular_esperanza_formula_con_vehiculos(ruta, asignacion_opt)
            fb_opt = None
            diff_opt = None
            skip_opt = None
            try:
                fb_opt = self.problem.calcular_esperanza_fuerza_bruta_con_vehiculos(ruta, asignacion_opt)
                diff_opt = abs(formula_opt - fb_opt)
            except ValueError as e:
                skip_opt = str(e)

            if vehiculos_ingresados is not None:
                # Costo determinista de TU asignacion -- ahora incluye el
                # costo de retorno por cada tramo contiguo (antes solo
                # sumaba arcos).
                costo_det_ing = self.problem.costo_determinista_con_asignacion(ruta, vehiculos_ingresados)

                formula_ing, _, _, _ = self.problem.calcular_esperanza_formula_con_vehiculos(ruta, vehiculos_ingresados)
                fb_ing = None
                diff_ing = None
                skip_ing = None
                try:
                    fb_ing = self.problem.calcular_esperanza_fuerza_bruta_con_vehiculos(ruta, vehiculos_ingresados)
                    diff_ing = abs(formula_ing - fb_ing)
                except ValueError as e:
                    skip_ing = str(e)

                diff_vs_opt = formula_ing - formula_opt
            else:
                costo_det_ing = None
                formula_ing = fb_ing = diff_ing = skip_ing = diff_vs_opt = None

            elapsed = time.time() - start_time

            self.lbl_ruta_evaluada.config(text=f"Ruta evaluada: {ruta}")

            texto_opt = f"Costo esperado (óptimo): fórmula={formula_opt:.6f}"
            if fb_opt is not None:
                texto_opt += f"  |  fuerza bruta={fb_opt:.6f}  (diff={diff_opt:.2e})"
            else:
                texto_opt += f"  |  fuerza bruta salteada ({skip_opt})"
            self.lbl_esperanza_opt.config(text=texto_opt)
            self.lbl_asignacion_opt.config(text=f"Asignación óptima de vehículos: {asignacion_opt}")
            self.lbl_costo_det_opt.config(text=f"Costo determinista (óptimo, arcos+retorno): {costo_det_opt:.6f}")

            if vehiculos_ingresados is not None:
                texto_ing = f"Costo esperado (ingresado): fórmula={formula_ing:.6f}"
                if fb_ing is not None:
                    texto_ing += f"  |  fuerza bruta={fb_ing:.6f}  (diff={diff_ing:.2e})"
                else:
                    texto_ing += f"  |  fuerza bruta salteada ({skip_ing})"
                self.lbl_esperanza_ing.config(text=texto_ing)
                self.lbl_costo_det_ing.config(text=f"Costo determinista (ingresado, arcos+retorno): {costo_det_ing:.6f}")
                self.lbl_diff.config(text=f"Diferencia esperado (ingresado - óptimo, por fórmula): {diff_vs_opt:.6f}")
            else:
                self.lbl_esperanza_ing.config(text="Costo esperado (ingresado): -- (no se ingresaron vehículos)")
                self.lbl_costo_det_ing.config(text="Costo determinista (ingresado): --")
                self.lbl_diff.config(text="Diferencia: --")

            self.lbl_tiempo_eval.config(text=f"Tiempo de cálculo: {elapsed:.4f} s")

        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular:\n{str(e)}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------
    # Métodos del Solver (con verificación añadida)
    # ------------------------------------------------------------
    def actualizar_mejor_solucion(self, iteracion, costo_deterministico, ruta, vehiculos, tiempo_mejor):
        def update():
            self.lbl_mejor_costo.config(text=f"Costo determinístico (ALNS, en progreso): {costo_deterministico:.6f}")
            self.lbl_mejor_iter.config(text=f"Iteración mejor solución: {iteracion}")
            self.lbl_mejor_tiempo.config(text=f"Tiempo mejor solución: {tiempo_mejor:.2f} s")
            if ruta and vehiculos:
                tramos = {}
                v_actual = vehiculos[0]
                inicio = 0
                for i, v in enumerate(vehiculos):
                    if v != v_actual:
                        tramos.setdefault(v_actual, []).append((inicio, i))
                        inicio = i
                        v_actual = v
                tramos.setdefault(v_actual, []).append((inicio, len(vehiculos)))
                texto_ruta = ""
                for v, segmentos in tramos.items():
                    for inicio, fin in segmentos:
                        ciudades = [ruta[inicio]] + [ruta[k+1] for k in range(inicio, fin)]
                        texto_ruta += f"Vehículo {v}: {' → '.join(map(str, ciudades))}  "
                self.lbl_mejor_ruta.config(text=f"Mejor ruta: {texto_ruta}")
                self.lbl_vehiculos_arco.config(text=f"Vehículos por arco (óptimos): {vehiculos}")
                self.lbl_tipos_usados.config(text=f"Tipos usados: {sorted(set(vehiculos))}")
            self.txt_result.insert(tk.END, f"✨ Mejora en iter {iteracion}: costo determinístico = {costo_deterministico:.6f}  (tiempo: {tiempo_mejor:.2f}s)\n")
            self.txt_result.see(tk.END)
        self.root.after(0, update)

    def actualizar_progreso(self, iteracion, costo):
        if self.tiempo_inicio:
            tiempo_total = time.time() - self.tiempo_inicio
            self.root.after(0, lambda: self.lbl_tiempo_total.config(text=f"Tiempo total de ejecución: {tiempo_total:.2f} s"))
            self.root.after(0, lambda: self.lbl_estado.config(text=f"Estado: Ejecutando (iter {iteracion})"))

    def exportar_desde_solver(self, ruta, vehiculos, costo_deterministico, iteracion, motivo):
        if self.problem and ruta:
            costo_v, v_arco, _, _ = self.problem.costo_ruta_con_vehiculos(ruta)
            costo_esp = self.solver.mejor_costo_esperado if self.solver else None
            costo_esp_fb = self.solver.mejor_costo_esperado_fb if self.solver else None
            diff_esp = self.solver.mejor_costo_esperado_diff if self.solver else None
            archivo = self.problem.exportar_solucion(
                ruta, v_arco, costo_esp, costo_v, iteracion, 1, motivo.lower(),
                costo_esperado_fb=costo_esp_fb, diff_esperado=diff_esp,
            )
            self.ultimo_archivo_exportado = archivo
            self.root.after(0, lambda: self.lbl_status.config(text=f"Exportado: {os.path.basename(archivo)}"))
            self.root.after(0, lambda: self.txt_result.insert(tk.END, f"\n📁 Exportado ({motivo}): {os.path.basename(archivo)}\n"))
            self.root.after(0, lambda: self.txt_result.see(tk.END))

    def ejecutar(self):
        if self.problem is None:
            messagebox.showwarning("Advertencia", "Primero cargue un archivo de instancia.")
            return
        try:
            max_iter = int(self.entry_iter.get())
            temp_ini = float(self.entry_temp.get())
            cooling = float(self.entry_cool.get())
            porc_destr = float(self.entry_destr.get())
            usar_estanc_iter = self.var_usar_estanc_iter.get()
            max_estanc_iter = int(self.entry_estanc_iter.get())
            usar_estanc_tiempo = self.var_usar_estanc_tiempo.get()
            max_estanc_tiempo = float(self.entry_estanc_tiempo.get())
        except ValueError:
            messagebox.showerror("Error", "Parámetros inválidos")
            return

        self.btn_run.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.DISABLED)
        self.txt_result.delete(1.0, tk.END)
        self.lbl_estado.config(text="Ejecutando...")
        self.lbl_mejor_costo.config(text="Costo esperado (ALNS): --")
        self.lbl_mejor_iter.config(text="Iteración mejor solución: --")
        self.lbl_mejor_tiempo.config(text="Tiempo mejor solución: --")
        self.lbl_mejor_ruta.config(text="Mejor ruta: --")
        self.lbl_vehiculos_arco.config(text="Vehículos por arco: --")
        self.lbl_tipos_usados.config(text="Tipos usados: --")
        self.lbl_tiempo_total.config(text="Tiempo total de ejecución: --")
        self.lbl_status.config(text="Estado: Esperando acción")
        self.lbl_verificacion.config(text="Verificación: --")

        self.tiempo_inicio = time.time()

        self.solver = ALNSSolver(
            problema=self.problem,
            max_iter=max_iter,
            temp_inicial=temp_ini,
            cooling=cooling,
            porc_destruccion=porc_destr,
            verbose=False,
            usar_estanc_iter=usar_estanc_iter,
            max_estanc_iter=max_estanc_iter,
            usar_estanc_tiempo=usar_estanc_tiempo,
            max_estanc_tiempo=max_estanc_tiempo
        )

        def run():
            mejor_ruta, mejor_costo = self.solver.ejecutar(
                tiempo_inicio=self.tiempo_inicio,
                callback_mejora=self.actualizar_mejor_solucion,
                callback_progreso=self.actualizar_progreso,
                export_callback=self.exportar_desde_solver
            )
            tiempo_total = time.time() - self.tiempo_inicio
            self.root.after(0, lambda: self.finalizar(mejor_ruta, mejor_costo, tiempo_total))

        self.hilo_ejecucion = threading.Thread(target=run)
        self.hilo_ejecucion.daemon = True
        self.hilo_ejecucion.start()

    def pausar(self):
        if self.solver:
            self.solver.pausar()
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_resume.config(state=tk.NORMAL)
            self.lbl_estado.config(text="PAUSADO")
            self.btn_export.config(state=tk.NORMAL)

    def reanudar(self):
        if self.solver:
            self.solver.reanudar()
            self.btn_pause.config(state=tk.NORMAL)
            self.btn_resume.config(state=tk.DISABLED)
            self.lbl_estado.config(text="Ejecutando...")
            self.btn_export.config(state=tk.DISABLED)

    def detener(self):
        if self.solver:
            self.solver.detener()
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_resume.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)
            self.lbl_estado.config(text="DETENIDO")
            self.btn_export.config(state=tk.NORMAL)
            tiempo_total = time.time() - self.tiempo_inicio
            self.lbl_tiempo_total.config(text=f"Tiempo total de ejecución: {tiempo_total:.2f} s")

    def exportar_ahora(self):
        if self.solver and self.problem:
            ruta = self.solver.mejor_ruta
            try:
                costo_v, v_arco, _, _ = self.problem.costo_ruta_con_vehiculos(ruta)
            except Exception:
                costo_v = self.solver.mejor_costo if self.solver else 0
                v_arco = []
            # Solo usa el costo esperado / fuerza bruta si ya se calcularon
            # (al final de la busqueda) -- no se recalculan aca para no
            # bloquear la UI con una fuerza bruta a mitad de corrida.
            costo_esp = self.solver.mejor_costo_esperado if self.solver else None
            costo_esp_fb = self.solver.mejor_costo_esperado_fb if self.solver else None
            diff_esp = self.solver.mejor_costo_esperado_diff if self.solver else None
            iteracion = self.solver.best_iteration if self.solver else 0
            if ruta:
                archivo = self.problem.exportar_solucion(
                    ruta, v_arco, costo_esp, costo_v, iteracion, 1, "manual",
                    costo_esperado_fb=costo_esp_fb, diff_esperado=diff_esp,
                )
                self.ultimo_archivo_exportado = archivo
                self.lbl_status.config(text=f"Exportado: {os.path.basename(archivo)}")
                self.txt_result.insert(tk.END, f"\n📁 Exportado manual: {os.path.basename(archivo)}\n")
                self.txt_result.see(tk.END)
                messagebox.showinfo("Exportación exitosa", f"Archivo guardado en:\n{archivo}")

    def finalizar(self, mejor_ruta, mejor_costo_det, tiempo_total):
        # El ALNS busco por costo DETERMINISTICO (mejor_costo_det). El
        # costo esperado (formula) y su validacion por fuerza bruta ya se
        # calcularon UNA SOLA VEZ dentro de ejecutar() -- se leen del
        # solver, no se recalculan aca (evita repetir una fuerza bruta
        # potencialmente cara, y evita el "self-check" circular de antes,
        # donde se comparaba el mismo numero contra si mismo).
        try:
            costo_det, vehiculos_opt, tipos, puntos = self.problem.costo_ruta_con_vehiculos(mejor_ruta)
        except Exception:
            vehiculos_opt = []
            tipos = set()
            puntos = []

        costo_esperado = self.solver.mejor_costo_esperado if self.solver else None
        costo_fb = self.solver.mejor_costo_esperado_fb if self.solver else None
        diff = self.solver.mejor_costo_esperado_diff if self.solver else None
        skip_reason = self.solver.mejor_costo_esperado_fb_skip_reason if self.solver else None

        self.btn_run.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.NORMAL)

        if diff is not None and diff < 1e-9:
            self.lbl_verificacion.config(text="✅ Verificación: fórmula del costo esperado coincide con fuerza bruta", foreground="green")
        elif diff is not None:
            self.lbl_verificacion.config(text=f"⚠️ Verificación: diferencia de {diff:.10f} entre fórmula y fuerza bruta", foreground="orange")
        elif skip_reason:
            self.lbl_verificacion.config(text=f"ℹ️ Validación por fuerza bruta salteada ({skip_reason})", foreground="gray")
        else:
            self.lbl_verificacion.config(text="❌ Verificación: no se pudo calcular", foreground="red")

        texto_costo = f"Costo determinístico (ALNS): {mejor_costo_det:.6f}"
        if costo_esperado is not None:
            texto_costo += f"  |  Costo esperado (PTSP): {costo_esperado:.6f}"
        self.lbl_mejor_costo.config(text=texto_costo)
        self.lbl_mejor_iter.config(text=f"Iteración mejor solución: {self.solver.best_iteration if self.solver else 0}")
        self.lbl_mejor_tiempo.config(text=f"Tiempo mejor solución: {self.solver.tiempo_mejor:.2f} s")
        self.lbl_tiempo_total.config(text=f"Tiempo total de ejecución: {tiempo_total:.2f} s")
        self.lbl_estado.config(text=f"Finalizado - Determinístico: {mejor_costo_det:.6f}")

        # Mostrar ruta
        if mejor_ruta and vehiculos_opt:
            tramos = {}
            v_actual = vehiculos_opt[0] if vehiculos_opt else None
            inicio = 0
            for i, v in enumerate(vehiculos_opt):
                if v != v_actual:
                    tramos.setdefault(v_actual, []).append((inicio, i))
                    inicio = i
                    v_actual = v
            if v_actual is not None:
                tramos.setdefault(v_actual, []).append((inicio, len(vehiculos_opt)))
            texto_ruta = ""
            for v, segmentos in tramos.items():
                for inicio, fin in segmentos:
                    ciudades = [mejor_ruta[inicio]] + [mejor_ruta[k+1] for k in range(inicio, fin)]
                    texto_ruta += f"Vehículo {v}: {' → '.join(map(str, ciudades))}  "
            self.lbl_mejor_ruta.config(text=f"Mejor ruta: {texto_ruta}")
            self.lbl_vehiculos_arco.config(text=f"Vehículos por arco (óptimos): {vehiculos_opt}")
            self.lbl_tipos_usados.config(text=f"Tipos usados: {sorted(tipos)}")
        else:
            self.lbl_mejor_ruta.config(text=f"Mejor ruta: {mejor_ruta}")

        # Log en el text area
        self.txt_result.insert(tk.END, "\n" + "="*60 + "\n")
        self.txt_result.insert(tk.END, "RESULTADOS FINALES\n")
        self.txt_result.insert(tk.END, f"Costo determinístico (ALNS, búsqueda): {mejor_costo_det:.6f}\n")
        if costo_esperado is not None:
            self.txt_result.insert(tk.END, f"Costo esperado (fórmula PTSP, sobre la ruta final): {costo_esperado:.6f}\n")
        if costo_fb is not None:
            self.txt_result.insert(tk.END, f"Costo esperado (fuerza bruta, validación): {costo_fb:.6f}\n")
            self.txt_result.insert(tk.END, f"Diferencia fórmula vs fuerza bruta: {diff:.10f}\n")
        elif skip_reason:
            self.txt_result.insert(tk.END, f"Validación por fuerza bruta salteada: {skip_reason}\n")
        self.txt_result.insert(tk.END, f"Iteración mejor solución: {self.solver.best_iteration if self.solver else 0}\n")
        self.txt_result.insert(tk.END, f"Tiempo mejor solución: {self.solver.tiempo_mejor:.2f} s\n")
        self.txt_result.insert(tk.END, f"Tiempo total: {tiempo_total:.2f} s\n")
        self.txt_result.insert(tk.END, f"Ruta: {mejor_ruta}\n")
        self.txt_result.insert(tk.END, f"Vehículos por arco (óptimos): {vehiculos_opt}\n")
        self.txt_result.insert(tk.END, f"Tipos usados: {sorted(tipos)}\n")
        if puntos:
            self.txt_result.insert(tk.END, f"Puntos de cambio: {puntos}\n")
        self.txt_result.see(tk.END)

        # Gráficos
        try:
            costo_v, _, _, _ = self.problem.costo_ruta_con_vehiculos(mejor_ruta)
            graficar_ruta_con_vehiculos(self.coordenadas, mejor_ruta, vehiculos_opt, tipos, costo_v, self.problem)
        except Exception as e:
            print("Error al graficar ruta con vehículos:", e)

        if costo_esperado is not None:
            try:
                graficar_ruta_esperanza(self.coordenadas, mejor_ruta, self.problem.probabilidades, costo_esperado, self.problem)
            except Exception as e:
                print("Error al graficar esperanza:", e)

        # Exportar si no se ha hecho
        if not self.ultimo_archivo_exportado and vehiculos_opt:
            archivo = self.problem.exportar_solucion(
                mejor_ruta, vehiculos_opt, costo_esperado,
                costo_det if 'costo_det' in locals() else mejor_costo_det,
                self.solver.best_iteration if self.solver else 0, 1, "final",
                costo_esperado_fb=costo_fb, diff_esperado=diff,
            )
            self.ultimo_archivo_exportado = archivo
            self.lbl_status.config(text=f"Exportado final: {os.path.basename(archivo)}")

        msg_esperado = f"Costo esperado (fórmula): {costo_esperado:.6f}\n" if costo_esperado is not None else ""
        msg_fb = (
            f"Fuerza bruta: {costo_fb:.6f}\n" if costo_fb is not None
            else (f"Fuerza bruta: salteada ({skip_reason})\n" if skip_reason else "")
        )
        msg_verif = "✅ Coinciden" if (diff is not None and diff < 1e-9) else ("⚠️ Diferencia" if diff is not None else "N/A")
        messagebox.showinfo("Completado", f"Ejecución finalizada.\n"
                                          f"Costo determinístico (ALNS): {mejor_costo_det:.6f}\n"
                                          f"{msg_esperado}"
                                          f"{msg_fb}"
                                          f"Verificación fórmula vs fuerza bruta: {msg_verif}\n"
                                          f"Iteración mejor: {self.solver.best_iteration if self.solver else 0}\n"
                                          f"Tiempo mejor: {self.solver.tiempo_mejor:.2f} s\n"
                                          f"Tiempo total: {tiempo_total:.2f} s\n"
                                          f"Archivo: {self.ultimo_archivo_exportado}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = CarsGUI()
    app.run()