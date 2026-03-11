---
title: "Greedy best-first"
---

| Notebook | Colab |
|---------|:-----:|
| Notebook 02 — Búsqueda informada (Greedy, Dijkstra, A\*) | <a href="https://colab.research.google.com/github/sonder-art/ia_p26/blob/main/clase/14_busqueda_informada/notebooks/02_busqueda_informada.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |

---

# Greedy best-first: seguir el instinto

> *"Greedy algorithms: at each step, take the best option available right now."*

---

Greedy es `busqueda_generica` con `frontera = ColaDePrioridad(key=h)`. La única diferencia respecto a BFS es el tipo de frontera: en lugar de una cola FIFO, usamos una cola de prioridad ordenada por $h(n)$ — la heurística. El algoritmo siempre expande el nodo que *parece* más cercano a la meta.

---

## 1. Intuición: ir directo al objetivo

**¿Para qué sirve Greedy?** Úsalo cuando necesitas una solución rápidamente y no te importa si es óptima. Greedy aprovecha todo el conocimiento del dominio encapsulado en $h(n)$ para moverse lo más directamente posible hacia la meta.

La estrategia es simple: en cada paso, expande el nodo que tiene el menor valor de $h(n)$ — el que está "más cerca" de la meta según nuestra estimación. No mira el pasado (cuánto costó llegar aquí), solo el futuro (cuánto cuesta llegar a la meta desde aquí).

Esto tiene una consecuencia inmediata: Greedy puede ser **muy rápido** cuando la heurística guía bien. En casos ideales (sin paredes, obstáculos, ni caminos engañosos), puede ir casi en línea recta desde el inicio hasta la meta.

Pero también tiene un problema grave: **Greedy es codicioso y cortoplacista**. Puede meterse en un callejón sin salida atractivo (bajo $h$) sin darse cuenta de que el verdadero camino óptimo requería alejarse brevemente de la meta antes de acercarse.

---

## 2. En lenguaje natural

La pregunta que Greedy responde es: *"¿cuál es el camino que parece más corto a la meta en cada paso?"* — no necesariamente el más barato ni el más corto real.

1. Crea una **cola de prioridad** con el nodo inicial (prioridad = $h(\text{inicio})$).
2. Mientras la cola no esté vacía:
   1. Saca el nodo con **menor $h(n)$** — el que parece más cercano a la meta.
   2. Si es la meta, reconstruye y devuelve el camino.
   3. Márcalo como explorado.
   4. Para cada vecino no explorado, calcula $h(\text{vecino})$ y añádelo a la cola.
3. Si la cola se vacía sin encontrar la meta, devuelve fallo.

La diferencia con BFS: el paso 2.1 saca el nodo con **menor $h(n)$** en lugar del más antiguo. La diferencia con A\*: **no se rastrea $g(n)$** — Greedy ignora completamente cuánto ha costado llegar hasta aquí.

---

## 3. Pseudocódigo

```
function GREEDY(problema, h):

    # ── Inicialización ─────────────────────────────────────────────────
    frontera ← new PriorityFrontier()
    frontera.push(problema.inicio, priority=h(problema.inicio))
    #   ↑ prioridad = h(n), NO g(n) — Greedy ignora el costo acumulado

    explorado ← empty set
    padre ← { problema.inicio: null }

    # ── Bucle principal ─────────────────────────────────────────────────
    while frontera is not empty:

        nodo ← frontera.pop()               # [P1] saca el nodo con MENOR h(n)
                                            #      ← única diferencia con BFS

        if problema.is_goal(nodo):
            return reconstruct(padre, nodo)

        explorado.add(nodo)

        for each (vecino, costo) in problema.neighbors(nodo):
            if vecino not in explorado and vecino not in frontera:
                padre[vecino] ← nodo
                frontera.push(vecino, priority=h(vecino))  # [P2] prioridad = h solo

    return FAILURE
```

**Diferencia mínima respecto a `GENERIC-SEARCH`**: solo cambia el tipo de frontera (`PriorityFrontier` con `key=h`) y la condición de vecinos no chequea si hay camino más barato (no hay $g$ que comparar).

### Versión con PriorityFrontier

```python
class PriorityFrontier:
    """Cola de prioridad — siempre saca el nodo con menor f(n)."""
    def __init__(self):
        self.heap = []          # (prioridad, nodo)
        self.miembros = {}      # nodo → prioridad actual

    def push(self, nodo, priority):
        heapq.heappush(self.heap, (priority, nodo))
        self.miembros[nodo] = priority

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


# Greedy = GENERIC-SEARCH con PriorityFrontier(key=h)
frontera = PriorityFrontier()
frontera.push(inicio, priority=h(inicio))
```

La tabla completa de instancias:

| Frontera | `pop()` devuelve | Algoritmo |
|---|---|---|
| `ColaDeFrontera` (FIFO) | el más antiguo | BFS |
| `PilaDeFrontera` (LIFO) | el más reciente | DFS |
| `PilaConLimite(d)` | el más reciente hasta prof. $d$ | IDDFS |
| `PriorityFrontier(key=h)` | el de menor $h(n)$ | **Greedy** |
| `PriorityFrontier(key=g)` | el de menor $g(n)$ | Dijkstra |
| `PriorityFrontier(key=g+h)` | el de menor $f(n)=g+h$ | A\* |

---

## 4. Ejemplo paso a paso

Grafo con 6 nodos. $h$ = distancia Manhattan al nodo Meta.

```
Posiciones (fila, col):
  Inicio=(0,0), A=(0,2), B=(2,0), C=(2,2), D=(1,3), Meta=(3,3)

Aristas (no ponderadas, solo para seguir el orden de expansión):
  Inicio→A, Inicio→B, A→C, A→D, B→C, C→Meta, D→Meta

h(Inicio)=6, h(A)=4, h(B)=4, h(C)=2, h(D)=2, h(Meta)=0
```

| Paso | Nodo expandido | $h(n)$ | Frontera tras expansión | Nota |
|:----:|---|:---:|---|---|
| 1 | Inicio | 6 | \{A:4, B:4\} | Expande ambos vecinos |
| 2 | A | 4 | \{B:4, C:2, D:2\} | Tie: A antes que B (orden de inserción) |
| 3 | C | 2 | \{B:4, D:2\} | C tiene menor $h$ |
| 4 | D | 2 | \{B:4, Meta:0\} | D también tiene $h=2$ |
| 5 | Meta | 0 | — | **¡Encontrada!** |

Camino devuelto: `Inicio → A → C → Meta`

**¿Es óptimo?** En este ejemplo sí (todas las aristas tienen el mismo costo). Pero ese fue un caso afortunado — en el siguiente ejemplo veremos que Greedy falla.

---

## 5. El caso de fallo: la pared engañosa

```
10×10 grid, Inicio=(1,1), Meta=(8,8)
Pared horizontal en fila 5, columnas 2–9
Único paso: columna 1 (izquierda)

La pared impide el camino directo.
El camino óptimo rodea por la izquierda.
```

Greedy ve que la meta está abajo-a-la-derecha y sigue expandiendo en esa dirección hasta chocar con la pared. Al no poder avanzar, retrocede pero ha expandido muchos nodos inútiles a la derecha, y el camino final es más largo que el óptimo.

![Greedy: caso fácil vs caso con pared]({{ '/14_busqueda_informada/images/04_greedy_success_failure.png' | url }})

El panel derecho muestra exactamente este escenario: Greedy expande nodos en la dirección equivocada, mientras A\* (en verde azulado) encuentra el rodeo óptimo por la izquierda directamente.

---

## 6. Implementación Python

```python
import heapq

def greedy(problema, h):
    """
    Greedy best-first.
    h: callable h(nodo) → estimación al objetivo.
    Devuelve (camino, nodos_expandidos) o (None, nodos_expandidos).
    """
    inicio = problema.inicio
    frontera = [(h(inicio), inicio)]   # min-heap por h(n)
    en_frontera = {inicio}
    explorado = set()
    padre = {inicio: None}
    nodos_expandidos = 0

    while frontera:
        _, nodo = heapq.heappop(frontera)
        if nodo not in en_frontera:    # entrada obsoleta (lazy deletion)
            continue
        en_frontera.discard(nodo)

        if problema.es_meta(nodo):
            return reconstruir_camino(padre, nodo), nodos_expandidos

        explorado.add(nodo)
        nodos_expandidos += 1

        for vecino, _ in problema.vecinos(nodo):   # ignora el costo
            if vecino not in explorado and vecino not in en_frontera:
                padre[vecino] = nodo
                heapq.heappush(frontera, (h(vecino), vecino))
                en_frontera.add(vecino)

    return None, nodos_expandidos
```

**Punto clave**: la línea `for vecino, _ in problema.vecinos(nodo)` ignora el costo de la arista (`_`). Greedy no necesita saber cuánto costó llegar aquí. Esa es la diferencia fundamental con Dijkstra y A\*.

---

## 7. Completitud y optimalidad

| Propiedad | Greedy | Condición |
|---|:---:|---|
| **Completo** | Sí* | En grafos finitos con conjunto `explorado` |
| **Óptimo** | **No** | Puede encontrar caminos subóptimos (ejemplo de la pared) |

*Sin conjunto `explorado`, Greedy puede entrar en ciclos infinitos.

**¿Por qué no es óptimo?** Porque ignora $g(n)$. Si un camino caro lleva a un nodo con bajo $h$, Greedy lo preferirá sobre un camino barato que lleva a un nodo con $h$ algo mayor — aunque el segundo sea globalmente mejor.

---

## 8. Complejidad de tiempo y espacio

### Tiempo

En el **peor caso**: $O(b^m)$ — igual que DFS. Si la heurística guía mal (o el problema tiene muchos callejones sin salida con bajo $h$), Greedy puede explorar todo el grafo.

En el **mejor caso** (heurística perfecta, $h = h^*$): $O(d)$ — expande solo los nodos del camino óptimo.

La diferencia entre estos extremos puede ser enorme. El comportamiento real depende completamente de la calidad de $h$.

### Espacio

$O(b^m)$ en el peor caso — igual que BFS. La frontera puede llenarse con todos los nodos descubiertos.

> Recordatorio: $b$ = factor de ramificación (máximo de vecinos), $d$ = prof. de la solución, $m$ = prof. máxima del grafo. Definidos en [03 — Algoritmo genérico →](../13_simple_search/03_busqueda_generica.md).

---

## 9. ¿Cuándo usar Greedy?

### Usa Greedy cuando:

| Señal del problema | ¿Por qué favorece Greedy? |
|---|---|
| **La velocidad importa más que la optimalidad** | Greedy no rastrea $g$ ni actualiza costos → más rápido |
| **Tienes una heurística muy buena** | Si $h \approx h^*$, Greedy va casi directamente a la meta |
| **El problema es de "buena-suficiente" solución** | NPC en videojuegos, prototipado rápido |
| **El espacio de estados es enorme** | Greedy explora menos que Dijkstra en casos favorables |

**Ejemplo real: NPC pathfinding en videojuegos.** Los personajes no jugadores necesitan encontrar rutas en tiempo real, a 60 fps, en mapas grandes. Una pequeña imprecisión en la ruta no importa visualmente. Greedy da rutas "suficientemente buenas" con mucho menos cómputo que A\*.

### Señales de alerta:

- El problema tiene paredes o zonas que parecen atractivas pero son callejones sin salida (Greedy se atasca).
- Necesitas garantía de camino óptimo (usa A\* o Dijkstra).
- La heurística es poco confiable o constante cero (Greedy degenera a DFS desordenado).

---

## 10. Resumen

| Propiedad | Valor | Justificación |
|---|---|---|
| Frontera | Cola de prioridad por $h(n)$ | Siempre expande el nodo "más cercano" a la meta |
| Tiempo | $O(b^m)$ peor / $O(d)$ mejor | Depende enteramente de la calidad de $h$ |
| Espacio | $O(b^m)$ peor | Frontera puede contener todos los nodos descubiertos |
| Completo | Sí (finito + explorado) | Sin explorado, puede ciclar |
| Óptimo | **No** | Ignora $g(n)$ — puede pagar demasiado |

> Recordatorio de notación: $b$ = factor de ramificación (**máximo** de vecinos; peor caso), $d$ = profundidad de la solución, $m$ = profundidad máxima del grafo. Definidos en [03 — Algoritmo genérico →](../13_simple_search/03_busqueda_generica.md).

---

**Siguiente:** [Dijkstra: el costo mínimo garantizado →](04_dijkstra.md)
