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
        """d^k_ij de la tesis (Silva 2011, p.44 item 3): costo de devolver
        el vehiculo v, ALQUILADO en ciudad_alquiler, ENTREGADO en
        ciudad_devolucion. Formula VECTOR (thesis p.53), igual que
        costo_arco_sin_retorno, generalizada a cualquier j (no solo el
        deposito) -- ver aco_cars_ptsp_exact_euclidean.py para el mismo fix."""
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
# 3. FUNCIONES DE GRAFICADO (RUTA CON VEHÍCULOS Y ESPERANZA)
# ------------------------------------------------------------
def graficar_ciudades(coordenadas, titulo):
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


def graficar_ruta_con_vehiculos(coordenadas, ruta, vehiculos_por_arco, tipos_usados, costo_total, problema, puntos_cambio):
    coords = np.array(coordenadas)
    colores = list(mcolors.TABLEAU_COLORS.values())
    color_por_vehiculo = {t: colores[t % len(colores)] for t in tipos_usados}

    retornos = []
    for dev, v, alq in puntos_cambio:
        retornos.append((alq, dev, v))

    costo_total_arcos = 0.0
    for idx in range(len(ruta)-1):
        i, j = ruta[idx], ruta[idx+1]
        v = vehiculos_por_arco[idx]
        if v is None:
            continue
        dist_ij = problema.dist[i][j]
        edge_i = problema.edge[v][i]
        edge_j = problema.edge[v][j]
        term_ew = (edge_i * 2 + edge_j * 3) / 3.0
        costo_arco = term_ew + dist_ij
        costo_total_arcos += costo_arco

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

        dist_ij = problema.dist[i][j]
        edge_i = problema.edge[v][i]
        edge_j = problema.edge[v][j]
        term_ew = (edge_i * 2 + edge_j * 3) / 3.0
        costo_arco = term_ew + dist_ij

        mid_x = (coords[i,0] + coords[j,0]) / 2
        mid_y = (coords[i,1] + coords[j,1]) / 2
        texto = f"{idx+1}\nd={dist_ij:.2f}\new={term_ew:.2f}\narc={costo_arco:.2f}"
        plt.text(mid_x, mid_y, texto, fontsize=7, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor=color),
                 color='black', zorder=5)

    from collections import defaultdict
    retornos_por_par = defaultdict(list)
    for alq, dev, v in retornos:
        if alq != dev:
            key = (min(alq, dev), max(alq, dev))
            retornos_por_par[key].append((alq, dev, v))

    offset_scale = 1.5
    for (ciudad1, ciudad2), lista_ret in retornos_por_par.items():
        n_ret = len(lista_ret)
        offsets = np.linspace(-offset_scale, offset_scale, n_ret) if n_ret > 1 else [0.0]
        ret_ida = [r for r in lista_ret if r[0] == ciudad1 and r[1] == ciudad2]
        ret_vuelta = [r for r in lista_ret if r[0] == ciudad2 and r[1] == ciudad1]
        lista_ordenada = []
        for i in range(max(len(ret_ida), len(ret_vuelta))):
            if i < len(ret_ida):
                lista_ordenada.append(ret_ida[i])
            if i < len(ret_vuelta):
                lista_ordenada.append(ret_vuelta[i])
        for (alq, dev, v), off in zip(lista_ordenada, offsets):
            x0, y0 = coords[alq, 0], coords[alq, 1]
            xd, yd = coords[dev, 0], coords[dev, 1]
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

            costo_ret = problema.costo_retorno(alq, dev, v)
            if alq != dev:
                termino_rr = (problema.return_rate[v][alq] * 2 + problema.return_rate[v][0] * 3) / 3.0
                distancia = problema.dist[alq][dev]
                texto_ret = f"RET v{v}: {costo_ret:.2f}\n(d={distancia:.2f} + rr={termino_rr:.2f})"
                tx = ux * 1.0 + px * off * 0.3
                ty = uy * 1.0 + py * off * 0.3
                label_x = xd + tx
                label_y = yd + ty
                plt.text(label_x, label_y, texto_ret, fontsize=8, ha='center', va='center',
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

    costo_total_retornos = sum(problema.costo_retorno(alq, dev, v) for alq, dev, v in retornos if alq != dev)

    titulo = f"RUTA (costo con vehículos: {costo_total:.2f})\n"
    titulo += f"{'='*60}\n"
    titulo += f"Suma de arcos: {costo_total_arcos:.2f}  |  Suma de retornos: {costo_total_retornos:.2f}\n"
    if retornos:
        titulo += "Retornos:\n" + "\n".join(f"  • Vehículo {v} → desde ciudad {alq} hasta ciudad {dev}: {problema.costo_retorno(alq, dev, v):.2f}"
                                            for alq, dev, v in retornos if alq != dev)
    else:
        titulo += "No hay costos de retorno."
    plt.title(titulo, fontsize=10, loc='left', fontfamily='monospace')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4. FUNCIONES PARA EL CÁLCULO DE ESPERANZA (FÓRMULA CORRECTA)
# ------------------------------------------------------------
def graficar_ruta_esperanza(coordenadas, ruta, probabilidades, valor_esperado, problema):
    coords = np.array(coordenadas)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_title(f"Valor esperado E[S] = {valor_esperado:.4f}", fontsize=14)

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
        (ver _calcular_expected_y_validar) -- no en cada evaluacion de la
        busqueda."""
        if len(ruta) < 2:
            return 0.0
        costo, vehiculos, _, _ = self.problema.costo_ruta_con_vehiculos(ruta)
        if not vehiculos:
            return float('inf')
        return costo

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
# 6. INTERFAZ GRÁFICA (COMPLETA)
# ------------------------------------------------------------
class CarsUnificadoGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CaRS Solver (OBJETIVO: ESPERANZA S-CARS)")
        self.root.geometry("950x900")
        self.problem = None
        self.solver = None
        self.coordenadas = None
        self.probabilidades = None
        self.ruta_actual = None
        self.vehiculos_actuales = None
        self.tipos_actuales = None
        self.puntos_cambio_actuales = None
        self.ultimo_archivo_exportado = None
        self.tiempo_inicio = 0
        self.var_mostrar_grafico = tk.BooleanVar(value=False)
        self.var_usar_estanc_iter = tk.BooleanVar(value=True)
        self.var_usar_estanc_tiempo = tk.BooleanVar(value=True)
        self.var_mostrar_vehiculos = tk.BooleanVar(value=False)  # desmarcado por defecto
        self.create_widgets()

    def create_widgets(self):
        # --- Frame 1: Carga ---
        frame_top = ttk.LabelFrame(self.root, text="1. Cargar instancia CaRS", padding=5)
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

        # --- Frame 2: Parámetros ALNS ---
        frame_params = ttk.LabelFrame(self.root, text="2. Parámetros ALNS", padding=5)
        frame_params.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame_params, text="Iteraciones:").grid(row=0, column=0, sticky=tk.W)
        self.entry_iter = ttk.Entry(frame_params, width=10)
        self.entry_iter.insert(0, "15000")
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

        # --- Estancamiento ---
        frame_estanc = ttk.LabelFrame(self.root, text="Criterios de estancamiento (opcionales)", padding=5)
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

        # --- Controles ---
        frame_controls = ttk.LabelFrame(self.root, text="3. Control", padding=5)
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
        self.btn_graficar_alns = ttk.Button(frame_controls, text="📊 Graficar ruta ALNS", command=self.graficar_ruta_alns, state=tk.DISABLED)
        self.btn_graficar_alns.pack(side=tk.LEFT, padx=5)

        # Checkbox para mostrar vehículos
        self.chk_mostrar_vehiculos = ttk.Checkbutton(
            frame_controls,
            text="📊 Mostrar gráfica de vehículos (costo determinístico)",
            variable=self.var_mostrar_vehiculos
        )
        self.chk_mostrar_vehiculos.pack(side=tk.LEFT, padx=5)

        # --- Información ---
        frame_info = ttk.LabelFrame(self.root, text="4. Mejor solución encontrada (objetivo: esperanza)", padding=5)
        frame_info.pack(fill=tk.X, padx=10, pady=5)
        self.lbl_mejor_costo = ttk.Label(frame_info, text="Costo esperado (S-CARS): --", font=("Arial", 10, "bold"))
        self.lbl_mejor_costo.pack(anchor=tk.W)
        self.lbl_mejor_iter = ttk.Label(frame_info, text="Iteración mejor solución: --")
        self.lbl_mejor_iter.pack(anchor=tk.W)
        self.lbl_mejor_tiempo = ttk.Label(frame_info, text="Tiempo mejor solución: --")
        self.lbl_mejor_tiempo.pack(anchor=tk.W)
        self.lbl_mejor_ruta = ttk.Label(frame_info, text="Mejor ruta: --", wraplength=800, justify=tk.LEFT)
        self.lbl_mejor_ruta.pack(anchor=tk.W)
        self.lbl_vehiculos_arco = ttk.Label(frame_info, text="Vehículos por arco (solo visual): --", wraplength=800, justify=tk.LEFT)
        self.lbl_vehiculos_arco.pack(anchor=tk.W)
        self.lbl_tipos_usados = ttk.Label(frame_info, text="Tipos usados (solo visual): --")
        self.lbl_tipos_usados.pack(anchor=tk.W)
        self.lbl_tiempo_total = ttk.Label(frame_info, text="Tiempo total de ejecución: --")
        self.lbl_tiempo_total.pack(anchor=tk.W)
        self.lbl_estado = ttk.Label(frame_info, text="Estado: Esperando")
        self.lbl_estado.pack(anchor=tk.W)

        # --- Log ---
        frame_result = ttk.LabelFrame(self.root, text="5. Log de mejoras (esperanza)", padding=5)
        frame_result.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.txt_result = tk.Text(frame_result, wrap=tk.WORD, height=15)
        scroll = ttk.Scrollbar(frame_result, orient=tk.VERTICAL, command=self.txt_result.yview)
        self.txt_result.configure(yscrollcommand=scroll.set)
        self.txt_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.lbl_status = ttk.Label(self.root, text="Estado: Esperando acción")
        self.lbl_status.pack(pady=5)

    # ==================== MÉTODOS DE LA INTERFAZ ====================
    def cargar_archivo(self):
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de instancia CaRS",
            initialdir="instances/euclidean",
            filetypes=[("Archivos CaRS", "*.car *.cars"), ("Todos los archivos", "*.*")]
        )
        if archivo:
            try:
                dist_matrix, edge_arr, return_arr, n, nv, coords, nombre, probs = cargar_instancia_cars(archivo)
                self.coordenadas = coords
                self.probabilidades = probs
                self.problem = CarRenterProblem(n, dist_matrix, edge_arr, return_arr)
                self.problem.nombre_instancia = nombre
                self.problem.ruta_instancia = archivo
                self.problem.probabilidades = probs
                self.problem.coordenadas = coords
                self.lbl_file.config(text=os.path.basename(archivo))
                messagebox.showinfo("Éxito", f"Instancia cargada: {n} ciudades, {nv} vehículos\nProbabilidades leídas: {len(probs)} valores")
                if self.var_mostrar_grafico.get():
                    graficar_ciudades(coords, f"Distribución - {os.path.basename(archivo)}")
                self.btn_run.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")

    def actualizar_mejor_solucion(self, iteracion, costo_deterministico, ruta, vehiculos, tiempo_mejor):
        def update():
            self.lbl_mejor_costo.config(text=f"Costo determinístico (ALNS, en progreso): {costo_deterministico:.6f}")
            self.lbl_mejor_iter.config(text=f"Iteración mejor solución: {iteracion}")
            self.lbl_mejor_tiempo.config(text=f"Tiempo mejor solución: {tiempo_mejor:.2f} s")
            if ruta:
                self.lbl_mejor_ruta.config(text=f"Mejor ruta: {ruta}")
                if self.problem:
                    costo_v, v_arco, tipos, puntos = self.problem.costo_ruta_con_vehiculos(ruta)
                    self.lbl_vehiculos_arco.config(text=f"Vehículos por arco (visual): {v_arco}")
                    self.lbl_tipos_usados.config(text=f"Tipos usados (visual): {sorted(tipos)}")
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
        self.btn_graficar_alns.config(state=tk.DISABLED)
        self.txt_result.delete(1.0, tk.END)
        self.lbl_estado.config(text="Ejecutando...")
        self.lbl_mejor_costo.config(text="Costo determinístico (ALNS, en progreso): --")
        self.lbl_mejor_iter.config(text="Iteración mejor solución: --")
        self.lbl_mejor_tiempo.config(text="Tiempo mejor solución: --")
        self.lbl_mejor_ruta.config(text="Mejor ruta: --")
        self.lbl_vehiculos_arco.config(text="Vehículos por arco (visual): --")
        self.lbl_tipos_usados.config(text="Tipos usados (visual): --")
        self.lbl_tiempo_total.config(text="Tiempo total de ejecución: --")
        self.lbl_status.config(text="Estado: Esperando acción")

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
            self.btn_graficar_alns.config(state=tk.NORMAL)
            tiempo_total = time.time() - self.tiempo_inicio
            self.lbl_tiempo_total.config(text=f"Tiempo total de ejecución: {tiempo_total:.2f} s")
            if self.solver.mejor_ruta:
                self.ruta_actual = self.solver.mejor_ruta
                _, vehic, tipos, puntos = self.problem.costo_ruta_con_vehiculos(self.ruta_actual)
                self.vehiculos_actuales = vehic
                self.tipos_actuales = tipos
                self.puntos_cambio_actuales = puntos

    def exportar_ahora(self):
        if self.solver and self.problem:
            ruta = self.solver.mejor_ruta
            # Solo usa el costo esperado / fuerza bruta si ya se calcularon
            # (al final de la busqueda) -- no se recalculan aca para no
            # bloquear la UI con una fuerza bruta a mitad de corrida.
            costo_esp = self.solver.mejor_costo_esperado
            costo_esp_fb = self.solver.mejor_costo_esperado_fb
            diff_esp = self.solver.mejor_costo_esperado_diff
            iteracion = self.solver.best_iteration
            if ruta:
                costo_v, v_arco, _, _ = self.problem.costo_ruta_con_vehiculos(ruta)
                archivo = self.problem.exportar_solucion(
                    ruta, v_arco, costo_esp, costo_v, iteracion, 1, "manual",
                    costo_esperado_fb=costo_esp_fb, diff_esperado=diff_esp,
                )
                self.ultimo_archivo_exportado = archivo
                self.lbl_status.config(text=f"Exportado: {os.path.basename(archivo)}")
                self.txt_result.insert(tk.END, f"\n📁 Exportado manual: {os.path.basename(archivo)}\n")
                self.txt_result.see(tk.END)
                messagebox.showinfo("Exportación exitosa", f"Archivo guardado en:\n{archivo}")

    def graficar_ruta_alns(self):
        if self.ruta_actual is None or self.vehiculos_actuales is None:
            messagebox.showwarning("Advertencia", "No hay una ruta disponible para graficar.")
            return
        if self.problem is None or self.coordenadas is None:
            messagebox.showwarning("Advertencia", "Faltan datos del problema.")
            return
        try:
            costo_v, _, _, _ = self.problem.costo_ruta_con_vehiculos(self.ruta_actual)
            graficar_ruta_con_vehiculos(
                self.coordenadas,
                self.ruta_actual,
                self.vehiculos_actuales,
                self.tipos_actuales,
                costo_v,
                self.problem,
                self.puntos_cambio_actuales or []
            )
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo graficar:\n{str(e)}")

    def finalizar(self, mejor_ruta, mejor_costo_det, tiempo_total):
        # El ALNS busco por costo DETERMINISTICO (mejor_costo_det). El
        # costo esperado (formula) y su validacion por fuerza bruta ya se
        # calcularon UNA SOLA VEZ dentro de ejecutar() -- se leen del
        # solver, no se recalculan aca (evita repetir una fuerza bruta
        # potencialmente cara, y evita comparar el mismo numero contra si
        # mismo como pasaba antes).
        _, vehiculos, tipos, puntos_cambio = self.problem.costo_ruta_con_vehiculos(mejor_ruta)
        self.ruta_actual = mejor_ruta
        self.vehiculos_actuales = vehiculos
        self.tipos_actuales = tipos
        self.puntos_cambio_actuales = puntos_cambio

        costo_esperado = self.solver.mejor_costo_esperado if self.solver else None
        costo_fb = self.solver.mejor_costo_esperado_fb if self.solver else None
        diff = self.solver.mejor_costo_esperado_diff if self.solver else None
        skip_reason = self.solver.mejor_costo_esperado_fb_skip_reason if self.solver else None

        self.btn_run.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.NORMAL)
        self.btn_graficar_alns.config(state=tk.NORMAL)
        self.lbl_estado.config(text=f"Finalizado - Determinístico: {mejor_costo_det:.6f}")
        self.lbl_tiempo_total.config(text=f"Tiempo total de ejecución: {tiempo_total:.2f} s")

        texto_costo = f"Costo determinístico (ALNS): {mejor_costo_det:.6f}"
        if costo_esperado is not None:
            texto_costo += f"  |  Costo esperado (PTSP): {costo_esperado:.6f}"
        self.lbl_mejor_costo.config(text=texto_costo)

        self.txt_result.insert(tk.END, "\n" + "="*60 + "\n")
        self.txt_result.insert(tk.END, "RESULTADOS FINALES\n")
        self.txt_result.insert(tk.END, f"Costo determinístico (ALNS, búsqueda): {mejor_costo_det:.6f}\n")
        if costo_esperado is not None:
            self.txt_result.insert(tk.END, f"Costo esperado (fórmula PTSP, sobre la ruta final): {costo_esperado:.6f}\n")
        if costo_fb is not None:
            self.txt_result.insert(tk.END, f"Costo esperado (fuerza bruta, validación): {costo_fb:.6f}\n")
            self.txt_result.insert(tk.END, f"Diferencia fórmula vs fuerza bruta: {diff:.10f}\n")
            self.txt_result.insert(tk.END, "Verificación: " + ("✅ coinciden" if diff < 1e-9 else "⚠️ diferencia") + "\n")
        elif skip_reason:
            self.txt_result.insert(tk.END, f"Validación por fuerza bruta salteada: {skip_reason}\n")
        self.txt_result.insert(tk.END, f"Iteración mejor solución: {self.solver.best_iteration if self.solver else 0}\n")
        self.txt_result.insert(tk.END, f"Tiempo mejor solución: {self.solver.tiempo_mejor:.2f} s\n")
        self.txt_result.insert(tk.END, f"Tiempo total: {tiempo_total:.2f} s\n")
        self.txt_result.insert(tk.END, f"Ruta: {mejor_ruta}\n")
        self.txt_result.insert(tk.END, f"Vehículos por arco: {vehiculos}\n")
        self.txt_result.insert(tk.END, f"Tipos usados: {sorted(tipos)}\n")
        if puntos_cambio:
            self.txt_result.insert(tk.END, f"Puntos de cambio: {puntos_cambio}\n")
        self.txt_result.see(tk.END)

        # --- Gráfica de vehículos (solo si el checkbox está marcado) ---
        if self.var_mostrar_vehiculos.get():
            try:
                costo_v, _, _, _ = self.problem.costo_ruta_con_vehiculos(mejor_ruta)
                graficar_ruta_con_vehiculos(
                    self.coordenadas,
                    mejor_ruta,
                    vehiculos,
                    tipos,
                    costo_v,
                    self.problem,
                    puntos_cambio
                )
            except Exception as e:
                print("Error al graficar ruta con vehículos:", e)

        # --- Gráfica de esperanza (siempre, si se pudo calcular) ---
        if costo_esperado is not None:
            try:
                graficar_ruta_esperanza(self.coordenadas, mejor_ruta, self.probabilidades, costo_esperado, self.problem)
            except Exception as e:
                print("Error al graficar esperanza:", e)

        if not self.ultimo_archivo_exportado:
            costo_v, v_arco, _, _ = self.problem.costo_ruta_con_vehiculos(mejor_ruta)
            archivo = self.problem.exportar_solucion(
                mejor_ruta, v_arco, costo_esperado, costo_v,
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


# ------------------------------------------------------------
# 7. EJECUCIÓN PRINCIPAL
# ------------------------------------------------------------
if __name__ == "__main__":
    app = CarsUnificadoGUI()
    app.run()