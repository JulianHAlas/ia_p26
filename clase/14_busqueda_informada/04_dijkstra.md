---
title: "Dijkstra"
---

| Notebook | Colab |
|---------|:-----:|
| Notebook 02 — Búsqueda informada (Greedy, Dijkstra, A\*) | <a href="https://colab.research.google.com/github/sonder-art/ia_p26/blob/main/clase/14_busqueda_informada/notebooks/02_busqueda_informada.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |

---

# Dijkstra: el costo mínimo garantizado

> *\"Dijkstra's algorithm solves the single-source shortest path problem on a graph with non-negative edge weights.\"*

---

Dijkstra es `busqueda_generica` con `frontera = ColaDePrioridad(key=g)`. Es A\* con $h(n) = 0$ para todo $n$ — no usa ninguna estimación hacia la meta, solo rastrea cuánto costó llegar aquí. A cambio, **garantiza el camino de menor costo** a todos los nodos alcanzables desde el inicio.

---

## 1. Intuición: inundación por costo

**¿Por qué Greedy no es suficiente?** Greedy va hacia donde parece más cercana la meta, pero puede meterse en caminos baratos hacia un nodo equivocado. Necesitamos un algoritmo que expanda siempre el nodo más barato disponible — sin importar dónde esté la meta.

La imagen mental es la de una **inundación por costo**: imagina verter agua en el nodo inicial. El agua se propaga por las aristas más baratas primero, luego las más caras. Cuando el agua llega a un nodo, lo hace por el camino de menor costo posible.

Esto tiene una propiedad poderosa: **la primera vez que sacamos un nodo de la cola de prioridad, hemos encontrado el camino óptimo hasta él**. No hay ningún camino alternativo más barato esperando ser descubierto, porque todos los nodos más baratos ya fueron procesados.

La consecuencia práctica: **Dijkstra calcula el árbol de rutas mínimas desde el inicio hacia todos los nodos alcanzables** — no solo hacia la meta. Esto lo hace ideal cuando necesitas responder múltiples consultas desde un mismo origen, o cuando no sabes de antemano cuál es la meta.

![Dijkstra como inundación por costo]({{ '/14_busqueda_informada/images/05_dijkstra_expansion.png' | url }})

---

## 2. El problema con BFS: costos desiguales

BFS garantiza el camino con menos aristas. Pero en grafos con pesos, \"menos aristas\" ≠ \"menor costo\":

```
Grafo ponderado:
    A ──1── B ──1── Meta
    │
    2
    │
    D ──1── Meta

BFS responde: A → D → Meta  (2 saltos, costo = 3)
Dijkstra:     A → B → Meta  (2 saltos, costo = 2) ← ¡más barato!
```

BFS falló porque contó aristas, no costos. Dijkstra usa una cola de prioridad para expandir siempre el nodo con menor $g(n)$ — el costo real acumulado desde el inicio.

---

## 3. En lenguaje natural

1. Inicializa $g(\text{inicio}) = 0$ y $g(n) = \infty$ para todos los demás nodos.
2. Crea una **cola de prioridad** con el nodo inicial (prioridad = $g(\text{inicio}) = 0$).
3. Mientras la cola no esté vacía:
   1. Saca el nodo con **menor $g(n)$** — el más barato hasta ahora.
   2. Si es la meta, reconstruye y devuelve el camino.
   3. Márcalo como explorado (ya encontramos su camino óptimo).
   4. Para cada vecino no explorado:
      - Calcula $g_{\text{nuevo}} = g(\text{nodo actual}) + \text{costo}(\text{nodo actual}, \text{vecino})$.
      - Si $g_{\text{nuevo}} < g(\text{vecino})$: **relaja** — actualiza $g(\text{vecino})$ y añade al cola.
4. Si la cola se vacía sin encontrar la meta, devuelve fallo.

**La operación clave es la «relajación»**: cuando encontramos un camino más barato a un vecino que el que teníamos hasta ahora, actualizamos su costo estimado. Esta es la única diferencia estructural con Greedy.

---

## 4. Pseudocódigo

```
function DIJKSTRA(problema):

    # ── Inicialización ─────────────────────────────────────────────────
    g ← dict con g[problema.inicio] = 0 y g[n] = ∞ para el resto
    frontera ← new PriorityFrontier()
    frontera.push(problema.inicio, priority=0)
    explorado ← empty set
    padre ← { problema.inicio: null }

    # ── Bucle principal ─────────────────────────────────────────────────
    while frontera is not empty:

        nodo ← frontera.pop()               # [P1] saca el nodo con MENOR g(n)

        if problema.is_goal(nodo):
            return reconstruct(padre, nodo)

        explorado.add(nodo)                 # [P2] primera vez = óptimo garantizado

        for each (vecino, costo) in problema.neighbors(nodo):
            if vecino not in explorado:
                g_nuevo = g[nodo] + costo

                if g_nuevo < g.get(vecino, ∞):    # [P3] RELAJACIÓN
                    g[vecino] = g_nuevo
                    padre[vecino] = nodo
                    frontera.push_or_update(vecino, priority=g_nuevo)

    return FAILURE
```

**Diferencias respecto a Greedy**:
1. `priority = g[nodo]` en lugar de `priority = h(nodo)` — prioridad por costo real, no estimación.
2. Se rastrea el diccionario `g` explícitamente.
3. `push_or_update` puede actualizar la prioridad de un vecino ya en la frontera (relajación).

### Versión con PriorityFrontier

```python
class PriorityFrontier:
    """Cola de prioridad con actualización de prioridad (lazy deletion)."""
    def __init__(self):
        self.heap = []          # (prioridad, nodo)
        self.miembros = {}      # nodo → prioridad actual

    def push(self, nodo, priority):
        heapq.heappush(self.heap, (priority, nodo))
        self.miembros[nodo] = priority

    def push_or_update(self, nodo, priority):
        """Si nodo ya está, actualiza su prioridad (lazy deletion)."""
        self.miembros[nodo] = priority           # siempre guarda la mejor
        heapq.heappush(self.heap, (priority, nodo))  # entrada nueva; la vieja se ignora al hacer pop

    def pop(self):
        while self.heap:
            pri, nodo = heapq.heappop(self.heap)
            if nodo in self.miembros and self.miembros[nodo] == pri:
                del self.miembros[nodo]
                return nodo
        return None

    def contains(self, nodo):
        return nodo in self.miembros

    def is_empty(self):
        return not self.miembros


# Dijkstra = GENERIC-SEARCH con PriorityFrontier(key=g)
g = {inicio: 0}
frontera = PriorityFrontier()
frontera.push(inicio, priority=0)
```

La tabla completa de instancias:

| Frontera | `pop()` devuelve | Algoritmo |
|---|---|---|
| `ColaDeFrontera` (FIFO) | el más antiguo | BFS |
| `PilaDeFrontera` (LIFO) | el más reciente | DFS |
| `PilaConLimite(d)` | el más reciente hasta prof. $d$ | IDDFS |
| `PriorityFrontier(key=h)` | el de menor $h(n)$ | Greedy |
| `PriorityFrontier(key=g)` | el de menor $g(n)$ | **Dijkstra** |
| `PriorityFrontier(key=g+h)` | el de menor $f(n)=g+h$ | A\* |

---

## 5. La relajación: qué es y por qué importa

La **relajación** de una arista $(u, v)$ con costo $c$ consiste en preguntar: *¿es el camino que pasa por $u$ hacia $v$ más barato que el mejor camino que conocemos a $v$ hasta ahora?*

```
Antes de relajar:  g[v] = 10   (camino antiguo)
Arista u → v:      g[u] = 3, costo = 4

g_nuevo = g[u] + costo = 3 + 4 = 7 < 10

Después de relajar: g[v] = 7   ← ¡actualización!
                    padre[v] = u
```

Cada relajación exitosa significa que encontramos un camino más barato a algún nodo. Dijkstra garantiza que cuando un nodo sale de la cola de prioridad, ya no puede ser relajado más — su $g(n)$ es definitivo.

---

## 6. Ejemplo paso a paso

Grafo con 6 nodos y pesos enteros. Inicio = A, Meta = F.

```
A ──2── B ──1── C
│       │       │
4       3       2
│       │       │
D ──1── E ──3── F

Pesos de aristas:
  A-B: 2   B-C: 1   C-F: 2
  A-D: 4   B-E: 3   E-F: 3
  D-E: 1
```

| Paso | Nodo expandido | $g(n)$ | Relajaciones realizadas | Frontera (g valores) |
|:----:|---|:---:|---|---|
| 1 | A | 0 | g[B]=2, g[D]=4 | {B:2, D:4} |
| 2 | B | 2 | g[C]=3, g[E]=5 | {C:3, D:4, E:5} |
| 3 | C | 3 | g[F]=5 | {D:4, E:5, F:5} |
| 4 | D | 4 | **g[E]=5→5** (sin cambio: 4+1=5 ≥ 5) | {E:5, F:5} |
| 5 | E | 5 | **g[F]=5→8?** No — 5+3=8 > 5, sin cambio | {F:5} |
| 6 | F | 5 | **¡Meta!** | — |

Camino devuelto: `A → B → C → F` con costo total **5**.

**¿Dónde ocurrió la relajación interesante?** En el paso 4, D intentó relajar E a 4+1=5, pero E ya tenía $g=5$ desde el paso 2 vía B. En grafos con pesos no negativos, el nodo que abre la cola de prioridad primero ya tiene su costo óptimo — por eso el paso 4 no mejoró a E.

---

## 7. ¿Por qué el conjunto explorado es válido?

Con Dijkstra, cuando sacamos un nodo de la cola de prioridad, su $g(n)$ es definitivo. La demostración informal:

> Si existiera un camino más barato a $n$, tendría que pasar por algún nodo $u$ con $g(u) < g(n)$ que todavía no hemos explorado. Pero si $g(u) < g(n)$, entonces $u$ habría salido de la cola antes que $n$, y ya habría sido explorado. Contradicción.

Este argumento requiere que **todos los pesos sean no negativos**. Con pesos negativos, añadir una arista podría hacer que un camino que parecía caro se vuelva el más barato — y el razonamiento anterior colapsa.

---

## 8. Implementación Python

```python
import heapq

def dijkstra(problema):
    """
    Dijkstra: camino de costo mínimo desde inicio hasta meta.
    Devuelve (camino, nodos_expandidos) o (None, nodos_expandidos).
    """
    inicio = problema.inicio
    g = {inicio: 0}
    frontera = [(0, inicio)]   # min-heap: (g(n), nodo)
    en_frontera = {inicio: 0}  # nodo → mejor g visto
    explorado = set()
    padre = {inicio: None}
    nodos_expandidos = 0

    while frontera:
        g_actual, nodo = heapq.heappop(frontera)

        if nodo not in en_frontera or en_frontera[nodo] != g_actual:
            continue   # entrada obsoleta (lazy deletion)
        del en_frontera[nodo]

        if problema.es_meta(nodo):
            return reconstruir_camino(padre, nodo), nodos_expandidos

        explorado.add(nodo)
        nodos_expandidos += 1

        for vecino, costo in problema.vecinos(nodo):   # usa el costo ← diferencia con Greedy
            if vecino not in explorado:
                g_nuevo = g_actual + costo
                if g_nuevo < g.get(vecino, float('inf')):
                    g[vecino] = g_nuevo
                    padre[vecino] = nodo
                    heapq.heappush(frontera, (g_nuevo, vecino))
                    en_frontera[vecino] = g_nuevo

    return None, nodos_expandidos
```

**Punto clave**: la línea `for vecino, costo in problema.vecinos(nodo)` **usa el costo de la arista** — al contrario que Greedy, que lo descartaba con `_`. Esa es la diferencia fundamental.

---

## 9. Completitud y optimalidad

| Propiedad | Dijkstra | Condición |
|---|:---:|---|
| **Completo** | Sí* | En grafos finitos con conjunto `explorado` |
| **Óptimo** | **Sí** | Si todos los pesos de arista son $\geq 0$ |

*Sin `explorado`, puede ciclar en grafos con ciclos.

**¿Por qué es óptimo?** Porque siempre expande el nodo con menor $g(n)$ conocido, y con pesos no negativos, la primera expansión ya es definitiva (ver sección 7).

---

## 10. Complejidad de tiempo y espacio

### Tiempo

Con un **binary heap** (como `heapq` de Python):

$$O((V + E) \log V)$$

- $V$ = número de vértices (nodos)
- $E$ = número de aristas
- Cada nodo entra y sale de la cola una vez: $O(V \log V)$
- Cada arista puede causar una relajación (inserción en heap): $O(E \log V)$

Con una **Fibonacci heap** (más compleja de implementar): $O(V \log V + E)$ — óptimo en teoría.

### Espacio

$O(V + E)$ — hay que almacenar el grafo, el diccionario $g$, la cola de prioridad, y el diccionario `padre`.

> Recordatorio: $V$ = nodos, $E$ = aristas. En grafos densos $E \approx V^2$, en grafos dispersos $E \approx V$.
>
> Para comparar con los algoritmos del módulo anterior: si $b$ = factor de ramificación y $d$ = profundidad de la solución, entonces $V \approx b^d$ y $E \approx V \cdot b$, por lo que la complejidad de Dijkstra en un árbol de búsqueda es $O(b^d \log b)$, similar a BFS pero con el $\log b$ por las operaciones de heap.

---

## 11. ¿Cuándo usar Dijkstra?

### Usa Dijkstra cuando:

| Señal del problema | ¿Por qué favorece Dijkstra? |
|---|---|
| **No tienes heurística** | Si $h(n) = 0$, A\* se convierte exactamente en Dijkstra |
| **Necesitas rutas a todos los nodos** | Dijkstra calcula el árbol completo de caminos mínimos |
| **El grafo tiene pesos variables** | BFS falla; Dijkstra garantiza optimalidad |
| **Los pesos son no negativos** | Condición necesaria para la garantía de optimalidad |

**Ejemplo real: enrutamiento de red.** Los routers usan versiones de Dijkstra (OSPF — Open Shortest Path First) para calcular las rutas de menor costo en una red. Cada router construye el árbol de caminos mínimos desde sí mismo hacia todos los demás routers. Las métricas de costo incluyen ancho de banda, latencia y confiabilidad.

### Dijkstra vs A\*:

| | Dijkstra | A\* |
|---|---|---|
| ¿Necesita heurística? | No | Sí |
| ¿Óptimo? | Sí | Sí (si $h$ es admisible) |
| ¿Expande todos los nodos? | Sí, en todas direcciones | No — se enfoca hacia la meta |
| **Mejor cuando** | Sin heurística disponible, o se necesitan rutas a todos los nodos | Hay una buena heurística y se busca un destino específico |

Si conoces la meta y puedes calcular una heurística admisible, A\* expandirá muchos menos nodos que Dijkstra.

### Señales de alerta:

- El grafo tiene pesos negativos → usa **Bellman-Ford** ($O(VE)$).
- Necesitas la ruta entre todos los pares de nodos → usa **Floyd-Warshall** ($O(V^3)$).
- Tienes una buena heurística y solo te importa un destino → usa **A\***.

---

## 12. Resumen

| Propiedad | Valor | Justificación |
|---|---|---|
| Frontera | Cola de prioridad por $g(n)$ | Siempre expande el nodo más barato hasta ahora |
| Tiempo | $O((V+E) \log V)$ | Binary heap; $V$ pops + $E$ inserciones |
| Espacio | $O(V + E)$ | Grafo + cola + diccionarios |
| Completo | Sí (finito + explorado) | Sin explorado, puede ciclar |
| Óptimo | **Sí** (pesos ≥ 0) | Primer pop = costo definitivo |

> **Conexión con A\***: Dijkstra es A\* con $h(n) = 0$ para todo $n$. Todo lo que aprenderemos sobre A\* (admisibilidad, consistencia, conjunto explorado) se hereda directamente de las propiedades que acabamos de demostrar para Dijkstra.

---

**Siguiente:** [A\*: lo mejor de los dos mundos →](05_a_estrella.md)
