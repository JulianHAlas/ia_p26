#!/usr/bin/env python3
"""
Laboratorio: Búsqueda Informada (imágenes para las notas)

Uso:
    cd clase/14_busqueda_informada
    python3 lab_informed_search.py

Genera ~12 imágenes en:
    clase/14_busqueda_informada/images/

Dependencias: numpy, matplotlib
"""

from pathlib import Path
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

COLORS = {
    "blue":   "#2E86AB",
    "red":    "#E94F37",
    "green":  "#27AE60",
    "gray":   "#7F8C8D",
    "orange": "#F39C12",
    "purple": "#8E44AD",
    "light":  "#ECF0F1",
    "dark":   "#2C3E50",
    "teal":   "#1ABC9C",
    "pink":   "#E91E8C",
}

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

np.random.seed(42)


def _save(fig, name: str) -> None:
    out = IMAGES_DIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  {out.name}")


# ---------------------------------------------------------------------------
# Search algorithm helpers (used to generate visualization data)
# ---------------------------------------------------------------------------

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(r, c, rows, cols, walls):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in walls:
            yield (nr, nc)


def _run_dijkstra(start, goal, rows, cols, walls, costs=None):
    """Returns (path, expansion_order, g_values).
    costs: dict {(r,c): cost_to_enter} — default 1 for all."""
    g = {start: 0}
    parent = {start: None}
    frontier = [(0, start)]
    explored = set()
    order = []

    while frontier:
        cost, node = heapq.heappop(frontier)
        if node in explored:
            continue
        explored.add(node)
        order.append(node)
        if node == goal:
            break
        r, c = node
        for nb in _neighbors(r, c, rows, cols, walls):
            step = (costs[nb] if costs and nb in costs else 1)
            new_g = g[node] + step
            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                parent[nb] = node
                heapq.heappush(frontier, (new_g, nb))

    # Reconstruct path
    path = []
    n = goal
    while n is not None:
        path.append(n)
        n = parent.get(n)
    path.reverse()
    return path if path[0] == start else [], order, g


def _run_greedy(start, goal, rows, cols, walls):
    """Returns (path, expansion_order). f(n) = h(n) only."""
    parent = {start: None}
    frontier = [(_manhattan(start, goal), start)]
    explored = set()
    order = []

    while frontier:
        _, node = heapq.heappop(frontier)
        if node in explored:
            continue
        explored.add(node)
        order.append(node)
        if node == goal:
            break
        r, c = node
        for nb in _neighbors(r, c, rows, cols, walls):
            if nb not in explored:
                parent[nb] = node
                heapq.heappush(frontier, (_manhattan(nb, goal), nb))

    path = []
    n = goal
    while n is not None:
        path.append(n)
        n = parent.get(n)
    path.reverse()
    return path if path and path[0] == start else [], order


def _run_astar(start, goal, rows, cols, walls, costs=None):
    """Returns (path, expansion_order). f(n) = g(n) + h(n)."""
    g = {start: 0}
    parent = {start: None}
    h0 = _manhattan(start, goal)
    frontier = [(h0, 0, start)]  # (f, g, node)
    explored = set()
    order = []

    while frontier:
        f, cost, node = heapq.heappop(frontier)
        if node in explored:
            continue
        explored.add(node)
        order.append(node)
        if node == goal:
            break
        r, c = node
        for nb in _neighbors(r, c, rows, cols, walls):
            step = (costs[nb] if costs and nb in costs else 1)
            new_g = g[node] + step
            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                parent[nb] = node
                h = _manhattan(nb, goal)
                heapq.heappush(frontier, (new_g + h, new_g, nb))

    path = []
    n = goal
    while n is not None:
        path.append(n)
        n = parent.get(n)
    path.reverse()
    return path if path and path[0] == start else [], order, g


def _draw_graph_weighted(ax, pos, edges_w, node_colors=None,
                         highlight_path=None, title="", node_labels=None):
    """Draw a small weighted graph on ax."""
    nodes = list(pos.keys())
    nc = node_colors or {}

    for u, v, w in edges_w:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        on_path = highlight_path and (
            (u, v) in zip(highlight_path, highlight_path[1:]) or
            (v, u) in zip(highlight_path, highlight_path[1:]))
        color = COLORS["red"] if on_path else COLORS["gray"]
        lw = 2.5 if on_path else 1.5
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.1, str(w), ha="center", va="center",
                fontsize=10, color=COLORS["dark"],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    for n in nodes:
        x, y = pos[n]
        color = nc.get(n, COLORS["blue"])
        circle = plt.Circle((x, y), 0.22, color=color, zorder=3)
        ax.add_patch(circle)
        label = node_labels.get(n, str(n)) if node_labels else str(n)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)

    ax.set_xlim(-0.5, max(x for x, _ in pos.values()) + 0.5)
    ax.set_ylim(-0.5, max(y for _, y in pos.values()) + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)


def _draw_grid_explored(ax, rows, cols, walls, explored_order, path, start, goal,
                        title="", color_explored="#AED6F1", color_path="#2E86AB",
                        show_order=False):
    """Draw a grid showing explored cells and the found path."""
    img = np.ones((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            if (r, c) in walls:
                img[r, c] = [0.2, 0.2, 0.2]  # wall: dark
            else:
                img[r, c] = [0.97, 0.97, 0.97]  # free: light gray

    # Explored (lighter shade)
    ec = mcolors.to_rgb(color_explored)
    for node in explored_order:
        if node not in walls and node != start and node != goal:
            img[node[0], node[1]] = ec

    # Path
    pc = mcolors.to_rgb(color_path)
    for node in path:
        if node != start and node != goal:
            img[node[0], node[1]] = pc

    # Start / Goal
    img[start[0], start[1]] = mcolors.to_rgb(COLORS["orange"])
    if goal in {n for n in explored_order} or goal in path:
        img[goal[0], goal[1]] = mcolors.to_rgb(COLORS["green"])
    else:
        img[goal[0], goal[1]] = mcolors.to_rgb(COLORS["gray"])

    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.text(start[1], start[0], "S", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")
    ax.text(goal[1], goal[0], "G", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")


# ---------------------------------------------------------------------------
# 1. Weighted graph + BFS failure
# ---------------------------------------------------------------------------
def plot_weighted_graph_and_bfs_failure() -> None:
    """Two panels: a weighted graph example, then BFS wrong vs optimal path."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Grafos con pesos: por qué BFS da la respuesta incorrecta",
                 fontsize=13, fontweight="bold", y=1.02)

    # Panel 1: simple weighted graph
    pos1 = {"A": (0, 1), "B": (1, 2), "C": (2, 1), "D": (1, 0)}
    edges1 = [("A", "B", 1), ("B", "C", 1), ("A", "D", 5), ("D", "C", 1)]
    _draw_graph_weighted(axes[0], pos1, edges1,
                         node_colors={"A": COLORS["orange"], "C": COLORS["green"]},
                         title="Grafo con pesos\n(cada arista tiene un costo)")

    # Panel 2: BFS wrong path vs optimal
    pos2 = {"A": (0, 1.5), "B": (1, 2.5), "C": (2, 2.5),
            "D": (1, 0.5), "Meta": (3, 1.5)}
    edges2 = [("A", "B", 1), ("B", "C", 1), ("C", "Meta", 1),
              ("A", "D", 2), ("D", "Meta", 1)]
    nc2 = {"A": COLORS["orange"], "Meta": COLORS["green"],
           "B": COLORS["red"], "C": COLORS["red"]}

    # Draw all edges first
    ax = axes[1]
    for u, v, w in edges2:
        x0, y0 = pos2[u]; x1, y1 = pos2[v]
        ax.plot([x0, x1], [y0, y1], color=COLORS["gray"], lw=1.5, zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.12, str(w), ha="center", va="center", fontsize=10,
                color=COLORS["dark"],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    # BFS path (wrong) in red dashes
    bfs_path = [("A", "B"), ("B", "C"), ("C", "Meta")]
    for u, v in bfs_path:
        x0, y0 = pos2[u]; x1, y1 = pos2[v]
        ax.plot([x0, x1], [y0, y1], color=COLORS["red"], lw=3,
                linestyle="--", zorder=2, alpha=0.8)

    # Optimal path in green solid
    opt_path = [("A", "D"), ("D", "Meta")]
    for u, v in opt_path:
        x0, y0 = pos2[u]; x1, y1 = pos2[v]
        ax.plot([x0, x1], [y0, y1], color=COLORS["green"], lw=3, zorder=2)

    for n, (x, y) in pos2.items():
        color = nc2.get(n, COLORS["blue"])
        circle = plt.Circle((x, y), 0.22, color=color, zorder=3)
        ax.add_patch(circle)
        lbl = n if n != "Meta" else "G"
        ax.text(x, y, lbl, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)

    ax.set_xlim(-0.5, 3.8)
    ax.set_ylim(-0.2, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("BFS elige el camino de menos saltos (costo=3)\nóptimo real tiene costo=3 — ¡pero por otra ruta!",
                 fontsize=10, fontweight="bold", pad=8)

    # Annotations
    axes[1].annotate("BFS: A→B→C→G\ncosto = 1+1+1 = 3 saltos, costo = 3",
                     xy=(0.5, 0.08), xycoords="axes fraction", fontsize=9,
                     color=COLORS["red"],
                     bbox=dict(boxstyle="round", fc="#FDEDEC", ec=COLORS["red"]))
    axes[1].annotate("Optimo: A->D->G\n2 saltos, costo = 2+1 = 3",
                     xy=(0.5, 0.22), xycoords="axes fraction", fontsize=9,
                     color=COLORS["green"],
                     bbox=dict(boxstyle="round", fc="#EAFAF1", ec=COLORS["green"]))

    fig.tight_layout()
    _save(fig, "01_weighted_graph_bfs_failure.png")


# ---------------------------------------------------------------------------
# 2. Priority queue vs FIFO vs LIFO
# ---------------------------------------------------------------------------
def plot_priority_queue_order() -> None:
    """Show how FIFO, LIFO, and priority queue order the same insertions differently."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Tres fronteras — mismos nodos, orden diferente",
                 fontsize=13, fontweight="bold", y=1.02)

    items = [("A", 3), ("B", 1), ("C", 5), ("D", 2), ("E", 4)]  # (label, priority/f-value)

    configs = [
        ("Cola FIFO\n(BFS)", [x[0] for x in items], COLORS["blue"],
         "Sale primero: A (llegó antes)"),
        ("Pila LIFO\n(DFS)", list(reversed([x[0] for x in items])), COLORS["orange"],
         "Sale primero: E (llegó después)"),
        ("Cola de prioridad\n(Dijkstra / A*)",
         [x[0] for x in sorted(items, key=lambda x: x[1])], COLORS["green"],
         "Sale primero: B (f=1, el menor)"),
    ]

    for ax, (title, order, color, note) in zip(axes, configs):
        y_pos = list(range(len(order)))
        colors_bar = [color] * len(order)
        colors_bar[0] = COLORS["red"]  # highlight the one that pops first

        bars = ax.barh(y_pos, [1] * len(order), color=colors_bar,
                       alpha=0.85, edgecolor="white", height=0.6)
        for i, (lbl, yp) in enumerate(zip(order, y_pos)):
            f_val = dict(items)[lbl]
            ax.text(0.5, yp, f"{lbl}  (f={f_val})", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(order) - 0.5)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.text(0.5, -0.7, note, ha="center", va="center",
                fontsize=10, color=COLORS["dark"],
                transform=ax.transData,
                bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["light"], ec=color, lw=1.5))

        # Arrow showing pop direction
        ax.annotate("", xy=(1.15, y_pos[0]), xytext=(1.05, y_pos[0]),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-|>", color=COLORS["red"], lw=2))
        ax.text(1.18, y_pos[0], "pop()", va="center", fontsize=9,
                color=COLORS["red"], fontweight="bold")

    fig.tight_layout()
    _save(fig, "02_priority_queue_order.png")


# ---------------------------------------------------------------------------
# 3. Heuristic spectrum
# ---------------------------------------------------------------------------
def plot_heuristic_spectrum() -> None:
    """Show the h(n) quality spectrum: h=0 vs h=loose vs h=Manhattan vs h=h*."""
    rows, cols = 12, 12
    start = (1, 1)
    goal = (10, 10)
    walls = set()
    # Add a few obstacles
    for c in range(3, 9):
        walls.add((5, c))
    walls.add((5, 9))

    heuristics = [
        ("h(n) = 0\n(Dijkstra)", lambda n: 0, "#AED6F1", "#2E86AB"),
        ("h(n) = Manhattan/2\n(admisible, débil)", lambda n: _manhattan(n, goal) / 2,
         "#A9DFBF", "#27AE60"),
        ("h(n) = Manhattan\n(admisible, fuerte)", lambda n: _manhattan(n, goal),
         "#F9E79F", "#F39C12"),
        ("h(n) ≈ h*(n)\n(casi perfecta)", lambda n: _manhattan(n, goal) * 0.95,
         "#F5CBA7", "#E94F37"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Calidad de la heurística: más ajustada → menos expansiones",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, (title, h_fn, c_exp, c_path) in zip(axes, heuristics):
        # Compute g values for drawing expansion intensity
        g = {start: 0}
        parent = {start: None}
        frontier = [(h_fn(start), 0, start)]
        explored = set()
        order = []

        while frontier:
            f, cost, node = heapq.heappop(frontier)
            if node in explored:
                continue
            explored.add(node)
            order.append(node)
            if node == goal:
                break
            r, c = node
            for nb in _neighbors(r, c, rows, cols, walls):
                new_g = g[node] + 1
                if nb not in g or new_g < g[nb]:
                    g[nb] = new_g
                    parent[nb] = node
                    heapq.heappush(frontier, (new_g + h_fn(nb), new_g, nb))

        path = []
        n = goal
        while n is not None:
            path.append(n)
            n = parent.get(n)
        path.reverse()

        img = np.ones((rows, cols, 3))
        for r in range(rows):
            for c in range(cols):
                if (r, c) in walls:
                    img[r, c] = [0.2, 0.2, 0.2]

        ec = mcolors.to_rgb(c_exp)
        for node in order:
            if node not in walls:
                img[node[0], node[1]] = ec

        pc = mcolors.to_rgb(c_path)
        for node in path:
            if node not in walls:
                img[node[0], node[1]] = pc

        img[start[0], start[1]] = mcolors.to_rgb(COLORS["orange"])
        img[goal[0], goal[1]] = mcolors.to_rgb(COLORS["green"])

        ax.imshow(img, interpolation="nearest", aspect="equal")
        ax.text(start[1], start[0], "S", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")
        ax.text(goal[1], goal[0], "G", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\n{len(order)} nodos expandidos",
                     fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, "03_heuristic_spectrum.png")


# ---------------------------------------------------------------------------
# 4. Greedy success and failure
# ---------------------------------------------------------------------------
def plot_greedy_success_and_failure() -> None:
    """Two panels: greedy succeeds (open grid) vs greedy fails (wall blocks direct path)."""
    rows, cols = 10, 10
    start = (1, 1)
    goal = (8, 8)

    # Panel 1: no obstacles — greedy succeeds
    walls1 = set()
    path1, order1 = _run_greedy(start, goal, rows, cols, walls1)
    path_opt1, order_opt1, _ = _run_astar(start, goal, rows, cols, walls1)

    # Panel 2: wall forces detour — greedy finds longer path
    walls2 = set()
    for c in range(2, 9):  # horizontal wall at row 5
        walls2.add((5, c))
    # gap at col 1 (left)
    walls2.discard((5, 1))
    # greedy will try to go right, hit wall, then find gap on right side
    # but optimal path goes through col 1 gap

    path2, order2 = _run_greedy(start, goal, rows, cols, walls2)
    path_opt2, _, _ = _run_astar(start, goal, rows, cols, walls2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Greedy best-first: rápido pero no siempre óptimo",
                 fontsize=13, fontweight="bold", y=1.02)

    _draw_grid_explored(axes[0], rows, cols, walls1, order1, path1, start, goal,
                        title=f"Sin obstáculos: Greedy OK\n{len(order1)} exp., camino={len(path1)-1} pasos",
                        color_explored="#D5E8D4", color_path="#27AE60")

    # For panel 2, show greedy path vs optimal
    img2 = np.ones((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            if (r, c) in walls2:
                img2[r, c] = [0.2, 0.2, 0.2]
            else:
                img2[r, c] = [0.97, 0.97, 0.97]

    # Explored by greedy
    for node in order2:
        if node not in walls2 and node != start and node != goal:
            img2[node[0], node[1]] = mcolors.to_rgb("#F9E79F")

    # Greedy path (orange)
    for node in path2:
        if node != start and node != goal and node not in walls2:
            img2[node[0], node[1]] = mcolors.to_rgb(COLORS["orange"])

    # Optimal path (green dashed)
    for node in path_opt2:
        if node != start and node != goal and node not in walls2:
            # blend with existing color if not on greedy path
            if node not in path2:
                img2[node[0], node[1]] = mcolors.to_rgb(COLORS["teal"])

    img2[start[0], start[1]] = mcolors.to_rgb(COLORS["orange"])
    img2[goal[0], goal[1]] = mcolors.to_rgb(COLORS["green"])

    axes[1].imshow(img2, interpolation="nearest", aspect="equal")
    axes[1].text(start[1], start[0], "S", ha="center", va="center",
                 fontsize=8, fontweight="bold", color="white")
    axes[1].text(goal[1], goal[0], "G", ha="center", va="center",
                 fontsize=8, fontweight="bold", color="white")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    opt_len = len(path_opt2) - 1 if path_opt2 else "?"
    greedy_len = len(path2) - 1 if path2 else "no encontrado"
    axes[1].set_title(
        f"Con pared: Greedy={greedy_len} pasos, Óptimo={opt_len} pasos\n"
        f"Naranja=Greedy, Verde azulado=Óptimo (A*)",
        fontsize=9, fontweight="bold")

    legend = [
        mpatches.Patch(color="#F9E79F", label=f"Expandido por Greedy ({len(order2)} nodos)"),
        mpatches.Patch(color=COLORS["orange"], label="Camino Greedy"),
        mpatches.Patch(color=COLORS["teal"], label="Camino óptimo (A*)"),
        mpatches.Patch(color=[0.2, 0.2, 0.2], label="Pared"),
    ]
    axes[1].legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "04_greedy_success_failure.png")


# ---------------------------------------------------------------------------
# 5. Dijkstra expansion (flood fill)
# ---------------------------------------------------------------------------
def plot_dijkstra_expansion() -> None:
    """Dijkstra flood-fill on a 15×15 grid with terrain costs."""
    rows, cols = 15, 15
    start = (1, 1)
    goal = (13, 13)

    walls = set()
    # L-shaped obstacle
    for c in range(4, 12):
        walls.add((6, c))
    for r in range(6, 11):
        walls.add((r, 11))

    # Terrain costs (default 1, rough terrain = 3)
    costs = {}
    for r in range(8, 12):
        for c in range(1, 5):
            costs[(r, c)] = 3

    path, order, g_vals = _run_dijkstra(start, goal, rows, cols, walls, costs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Dijkstra: expansión por costo mínimo acumulado g(n)",
                 fontsize=13, fontweight="bold", y=1.02)

    # Panel 1: expansion order colored by order (lighter = earlier)
    img1 = np.ones((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            if (r, c) in walls:
                img1[r, c] = [0.2, 0.2, 0.2]
            elif (r, c) in costs and costs[(r, c)] > 1:
                img1[r, c] = [0.93, 0.85, 0.70]  # light brown = rough terrain
            else:
                img1[r, c] = [0.97, 0.97, 0.97]

    cmap_exp = plt.cm.Blues
    n_exp = len(order)
    for i, node in enumerate(order):
        if node not in walls and node != start and node != goal:
            intensity = 0.25 + 0.65 * (i / max(n_exp - 1, 1))
            img1[node[0], node[1]] = cmap_exp(intensity)[:3]

    for node in path:
        if node != start and node != goal:
            img1[node[0], node[1]] = mcolors.to_rgb(COLORS["red"])
    img1[start[0], start[1]] = mcolors.to_rgb(COLORS["orange"])
    img1[goal[0], goal[1]] = mcolors.to_rgb(COLORS["green"])

    axes[0].imshow(img1, interpolation="nearest", aspect="equal")
    axes[0].text(start[1], start[0], "S", ha="center", va="center",
                 fontsize=7, fontweight="bold", color="white")
    axes[0].text(goal[1], goal[0], "G", ha="center", va="center",
                 fontsize=7, fontweight="bold", color="white")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title(f"Orden de expansión (azul oscuro = expandido después)\n"
                      f"{len(order)} nodos expandidos, camino óptimo en rojo",
                      fontsize=9, fontweight="bold")

    # Panel 2: g(n) heat map
    g_grid = np.full((rows, cols), np.nan)
    for (r, c), val in g_vals.items():
        if (r, c) not in walls:
            g_grid[r, c] = val

    masked = np.ma.masked_invalid(g_grid)
    wall_img = np.zeros((rows, cols))
    for (r, c) in walls:
        wall_img[r, c] = 1

    axes[1].imshow(masked, cmap="YlOrRd", interpolation="nearest", aspect="equal")
    axes[1].imshow(wall_img, cmap="Greys", alpha=0.5, interpolation="nearest",
                   aspect="equal", vmin=0, vmax=2)
    axes[1].text(start[1], start[0], "S", ha="center", va="center",
                 fontsize=7, fontweight="bold", color="white")
    axes[1].text(goal[1], goal[0], "G", ha="center", va="center",
                 fontsize=7, fontweight="bold", color="black")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Valores g(n) = costo real desde inicio\n(zonas marrones: terreno costoso)",
                      fontsize=9, fontweight="bold")
    cb = fig.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label("g(n)", fontsize=9)

    fig.tight_layout()
    _save(fig, "05_dijkstra_expansion.png")


# ---------------------------------------------------------------------------
# 6. A* expansion
# ---------------------------------------------------------------------------
def plot_a_star_expansion() -> None:
    """A* focused beam on same grid as Dijkstra."""
    rows, cols = 15, 15
    start = (1, 1)
    goal = (13, 13)

    walls = set()
    for c in range(4, 12):
        walls.add((6, c))
    for r in range(6, 11):
        walls.add((r, 11))

    path, order, _ = _run_astar(start, goal, rows, cols, walls)
    path_d, order_d, _ = _run_dijkstra(start, goal, rows, cols, walls)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("A* vs Dijkstra: misma garantía de optimalidad, muchos menos nodos",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, (o, p, title, c_exp) in zip(axes, [
        (order_d, path_d, f"Dijkstra: {len(order_d)} nodos expandidos", "#AED6F1"),
        (order,   path,   f"A* (h=Manhattan): {len(order)} nodos expandidos", "#A9DFBF"),
    ]):
        _draw_grid_explored(ax, rows, cols, walls, o, p, start, goal,
                            title=title, color_explored=c_exp,
                            color_path=COLORS["red"] if "Dijkstra" in title else COLORS["green"])

    fig.tight_layout()
    _save(fig, "06_a_star_vs_dijkstra.png")


# ---------------------------------------------------------------------------
# 7. Three-way comparison: Greedy / Dijkstra / A*  — CENTERPIECE
# ---------------------------------------------------------------------------
def plot_three_way_comparison() -> None:
    """3-panel: Greedy / Dijkstra / A* on 20×20 grid with horizontal wall + left gap."""
    rows, cols = 20, 20
    start = (2, 2)
    goal = (17, 17)

    walls = set()
    # Horizontal wall at row 10, gap on the left (cols 0-2)
    for c in range(3, 20):
        walls.add((10, c))

    path_g,  order_g  = _run_greedy(start, goal, rows, cols, walls)
    path_d,  order_d, _ = _run_dijkstra(start, goal, rows, cols, walls)
    path_a,  order_a, _ = _run_astar(start, goal, rows, cols, walls)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Greedy / Dijkstra / A* — mismo problema, estrategias distintas",
                 fontsize=13, fontweight="bold", y=1.02)

    configs = [
        (order_g, path_g, "Greedy (f = h)",
         "#F9E79F", COLORS["orange"], "[NO optimo]"),
        (order_d, path_d, "Dijkstra (f = g)",
         "#AED6F1", COLORS["blue"], "[Optimo]"),
        (order_a, path_a, "A* (f = g + h)",
         "#A9DFBF", COLORS["green"], "[Optimo]"),
    ]

    for ax, (order, path, name, c_exp, c_path, opt_tag) in zip(axes, configs):
        img = np.ones((rows, cols, 3))
        for r in range(rows):
            for c in range(cols):
                if (r, c) in walls:
                    img[r, c] = [0.2, 0.2, 0.2]
                else:
                    img[r, c] = [0.97, 0.97, 0.97]

        ec = mcolors.to_rgb(c_exp)
        for node in order:
            if node not in walls and node != start and node != goal:
                img[node[0], node[1]] = ec

        pc = mcolors.to_rgb(c_path)
        for node in path:
            if node != start and node != goal and node not in walls:
                img[node[0], node[1]] = pc

        img[start[0], start[1]] = mcolors.to_rgb(COLORS["orange"])
        img[goal[0], goal[1]] = mcolors.to_rgb(COLORS["green"])

        ax.imshow(img, interpolation="nearest", aspect="equal")
        ax.text(start[1], start[0], "S", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")
        ax.text(goal[1], goal[0], "G", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")
        ax.set_xticks([])
        ax.set_yticks([])

        path_len = len(path) - 1 if path else "–"
        ax.set_title(f"{name}\n{len(order)} nodos exp. · camino={path_len} · {opt_tag}",
                     fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, "07_three_way_comparison.png")


# ---------------------------------------------------------------------------
# 8. Heuristic quality effect on A*
# ---------------------------------------------------------------------------
def plot_heuristic_quality_effect() -> None:
    """Bar chart: nodes expanded by A* under different h quality levels."""
    # Use a solvable 8-puzzle instance solved with different heuristics
    # Simulate with the grid problem instead for visual clarity
    rows, cols = 15, 15
    start = (1, 1)
    goal = (13, 13)

    walls = set()
    for c in range(3, 12):
        walls.add((7, c))

    heuristics = {
        "h=0\n(Dijkstra)": lambda n: 0,
        "h = Man./2\n(débil)": lambda n: _manhattan(n, goal) / 2,
        "h = Manhattan\n(buena)": lambda n: _manhattan(n, goal),
        "h = Man.×0.95\n(casi perfecta)": lambda n: _manhattan(n, goal) * 0.95,
    }

    results = {}
    for name, h_fn in heuristics.items():
        g = {start: 0}
        parent = {start: None}
        frontier = [(h_fn(start), 0, start)]
        explored = set()
        count = 0

        while frontier:
            f, cost, node = heapq.heappop(frontier)
            if node in explored:
                continue
            explored.add(node)
            count += 1
            if node == goal:
                break
            r, c = node
            for nb in _neighbors(r, c, rows, cols, walls):
                new_g = g[node] + 1
                if nb not in g or new_g < g[nb]:
                    g[nb] = new_g
                    parent[nb] = node
                    heapq.heappush(frontier, (new_g + h_fn(nb), new_g, nb))

        results[name] = count

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    counts = list(results.values())
    bar_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["teal"]]

    bars = ax.bar(names, counts, color=bar_colors, alpha=0.85,
                  edgecolor="white", width=0.55)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("Nodos expandidos", fontsize=12)
    ax.set_title("Calidad de la heurística → eficiencia de A*\n"
                 "Misma garantía de optimalidad, menos trabajo con h mejor",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.tick_params(axis="x", labelsize=10)

    ax.annotate("← mismo resultado\nóptimo en todos",
                xy=(0.5, 0.7), xycoords="axes fraction", fontsize=10,
                ha="center", color=COLORS["gray"])

    fig.tight_layout()
    _save(fig, "08_heuristic_quality_effect.png")


# ---------------------------------------------------------------------------
# 9. Relaxed problem illustration (8-puzzle)
# ---------------------------------------------------------------------------
def plot_relaxed_problem_8puzzle() -> None:
    """Illustrate relaxed problem technique for 8-puzzle heuristics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Técnica del problema relajado: origen de las heurísticas admisibles",
                 fontsize=13, fontweight="bold", y=1.02)

    # State: 3×3 grid showing a specific 8-puzzle configuration
    # State: [[1,2,3],[4,0,6],[7,5,8]]  goal: [[1,2,3],[4,5,6],[7,8,0]]
    state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    def draw_puzzle(ax, grid, title, highlight=None):
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(-0.1, 3.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        for r in range(3):
            for c in range(3):
                val = grid[r][c]
                color = (COLORS["light"] if val == 0 else
                         COLORS["orange"] if (highlight and (r, c) in highlight) else
                         COLORS["blue"])
                rect = plt.Rectangle((c, 2 - r), 1, 1, color=color,
                                     ec="white", lw=2)
                ax.add_patch(rect)
                if val != 0:
                    ax.text(c + 0.5, 2 - r + 0.5, str(val),
                            ha="center", va="center",
                            fontsize=16, fontweight="bold",
                            color="white" if color != COLORS["light"] else COLORS["dark"])

    # Find misplaced tiles
    misplaced = set()
    for r in range(3):
        for c in range(3):
            if state[r][c] != 0 and state[r][c] != goal_state[r][c]:
                misplaced.add((r, c))

    draw_puzzle(axes[0], state,
                "Estado actual\n(fichas naranja = fuera de lugar)",
                highlight=misplaced)
    axes[0].text(1.5, -0.3,
                 f"h1 = fichas fuera de lugar = {len(misplaced)}\n"
                 f"Relajación: cada ficha puede moverse a cualquier lugar",
                 ha="center", fontsize=9, color=COLORS["dark"],
                 bbox=dict(boxstyle="round", fc="#FEF9E7", ec=COLORS["orange"]))

    # Manhattan distance
    draw_puzzle(axes[1], state,
                "h2 = distancia Manhattan\nRelajación: fichas se mueven sin bloquearse",
                highlight=misplaced)

    # Compute manhattan distance for display
    goal_pos = {}
    for r in range(3):
        for c in range(3):
            goal_pos[goal_state[r][c]] = (r, c)

    total_man = 0
    for r in range(3):
        for c in range(3):
            val = state[r][c]
            if val != 0:
                gr, gc = goal_pos[val]
                d = abs(r - gr) + abs(c - gc)
                if d > 0:
                    axes[1].annotate("", xy=(goal_pos[val][1] + 0.5, 2 - goal_pos[val][0] + 0.5),
                                     xytext=(c + 0.5, 2 - r + 0.5),
                                     arrowprops=dict(arrowstyle="-|>",
                                                     color=COLORS["teal"],
                                                     lw=1.2, alpha=0.6))
                    total_man += d

    axes[1].text(1.5, -0.3,
                 f"h2 = Σ distancias Manhattan = {total_man}\n"
                 f"h2 ≥ h1 siempre → Manhattan domina fichas fuera de lugar",
                 ha="center", fontsize=9, color=COLORS["dark"],
                 bbox=dict(boxstyle="round", fc="#EBF5FB", ec=COLORS["teal"]))

    draw_puzzle(axes[2], goal_state,
                "Estado meta\n(referencia)",
                highlight=None)
    axes[2].text(1.5, -0.3,
                 "h* = costo real (desconocido)\n"
                 "h1 ≤ h2 ≤ h* → ambas admisibles\n"
                 "h2 domina h1: A* con Manhattan expande menos nodos",
                 ha="center", fontsize=9, color=COLORS["dark"],
                 bbox=dict(boxstyle="round", fc="#E9F7EF", ec=COLORS["green"]))

    fig.tight_layout()
    _save(fig, "09_relaxed_problem_8puzzle.png")


# ---------------------------------------------------------------------------
# 10. IDA* iterations trace
# ---------------------------------------------------------------------------
def plot_ida_star_iterations() -> None:
    """3-panel: IDA* on the 5-node example, f_limit iterations."""
    # Graph: Start--(2)--A--(3)--Goal, Start--(4)--B--(1)--Goal
    # h: Start=3, A=2, B=1, Goal=0
    # f values: Start=0+3=3, A=2+2=4, B=4+1=5, Goal via A=5+0=5

    h_vals = {"Start": 3, "A": 2, "B": 1, "Goal": 0}
    edges = {"Start": [("A", 2), ("B", 4)], "A": [("Goal", 3)], "B": [("Goal", 1)], "Goal": []}
    pos = {"Start": (0, 1.5), "A": (2, 2.8), "B": (2, 0.2), "Goal": (4, 1.5)}

    def compute_f(node, g):
        return g + h_vals.get(node, 0)

    # Iteration traces
    iterations = [
        {
            "limit": 3,
            "label": "Iteración 1: f_limit = h(Start) = 3",
            "visited": ["Start"],
            "pruned": [("A", 4, "f=4>3"), ("B", 5, "f=5>3")],
            "next_limit": 4,
            "found": False,
        },
        {
            "limit": 4,
            "label": "Iteración 2: f_limit = 4",
            "visited": ["Start", "A"],
            "pruned": [("B", 5, "f=5>4"), ("Goal", 5, "f=5>4")],
            "next_limit": 5,
            "found": False,
        },
        {
            "limit": 5,
            "label": "Iteración 3: f_limit = 5  →  SOLUCIÓN ENCONTRADA",
            "visited": ["Start", "A", "Goal"],
            "pruned": [],
            "next_limit": None,
            "found": True,
            "path": ["Start", "A", "Goal"],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("IDA*: DFS con límite de f(n) = g(n) + h(n), creciente por iteración",
                 fontsize=13, fontweight="bold", y=1.02)

    all_edges_w = []
    for u, neighbors in edges.items():
        for v, w in neighbors:
            all_edges_w.append((u, v, w))

    for ax, it in zip(axes, iterations):
        visited_set = set(it["visited"])
        pruned_set = {p[0] for p in it["pruned"]}
        path_set = set(it.get("path", []))

        nc = {}
        for n in pos:
            if n in path_set:
                nc[n] = COLORS["green"]
            elif n in visited_set:
                nc[n] = COLORS["blue"]
            elif n in pruned_set:
                nc[n] = COLORS["red"]
            else:
                nc[n] = COLORS["gray"]

        nc["Start"] = COLORS["orange"]
        nc["Goal"] = COLORS["green"] if "Goal" in visited_set else COLORS["gray"]

        highlight_path = it.get("path", [])
        _draw_graph_weighted(ax, pos, all_edges_w, node_colors=nc,
                             highlight_path=highlight_path,
                             title=it["label"])

        # Add f-value annotations
        for n, (x, y) in pos.items():
            if n in visited_set or n in pruned_set:
                g_approx = {"Start": 0, "A": 2, "B": 4, "Goal": 5}
                g_val = g_approx.get(n, "?")
                f_val = compute_f(n, g_val if isinstance(g_val, (int, float)) else 0)
                color = COLORS["red"] if n in pruned_set else COLORS["dark"]
                label = f"f={f_val}" + (" [X]" if n in pruned_set else "")
                ax.text(x, y - 0.38, label, ha="center", va="top",
                        fontsize=8, color=color, fontweight="bold")

        # Bottom annotation
        if it["pruned"]:
            pruned_str = ", ".join(f"{p[0]}({p[2]})" for p in it["pruned"])
            next_str = f"→ próx. f_limit = {it['next_limit']}"
            ax.text(2, -0.1, f"Podado: {pruned_str}\n{next_str}",
                    ha="center", va="top", fontsize=8.5,
                    color=COLORS["dark"],
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FDFEFE", ec=COLORS["gray"]))
        else:
            ax.text(2, -0.1, "¡Solución: Start→A→Goal, costo=5!",
                    ha="center", va="top", fontsize=9,
                    color=COLORS["green"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#EAFAF1", ec=COLORS["green"]))

        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 3.8)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS["orange"], label="Start"),
        mpatches.Patch(color=COLORS["blue"], label="Visitado"),
        mpatches.Patch(color=COLORS["red"], label="Podado (f > límite)"),
        mpatches.Patch(color=COLORS["green"], label="Solución"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    _save(fig, "10_ida_star_iterations.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generando imágenes — Búsqueda Informada")
    print("=" * 45)

    # 1. Weighted graph + BFS failure
    plot_weighted_graph_and_bfs_failure()

    # 2. Priority queue order
    plot_priority_queue_order()

    # 3. Heuristic spectrum
    plot_heuristic_spectrum()

    # 4. Greedy success and failure
    plot_greedy_success_and_failure()

    # 5. Dijkstra expansion
    plot_dijkstra_expansion()

    # 6. A* vs Dijkstra
    plot_a_star_expansion()

    # 7. Three-way comparison — CENTERPIECE
    plot_three_way_comparison()

    # 8. Heuristic quality effect
    plot_heuristic_quality_effect()

    # 9. Relaxed problem (8-puzzle)
    plot_relaxed_problem_8puzzle()

    # 10. IDA* iterations
    plot_ida_star_iterations()

    print("=" * 45)
    print(f"✓  Todas las imágenes guardadas en {IMAGES_DIR}")


if __name__ == "__main__":
    main()
