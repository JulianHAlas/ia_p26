---
title: "Poda alfa-beta"
---

| Notebook | Colab |
|---------|:-----:|
| Notebook 02 — Minimax y alpha-beta | <a href="https://colab.research.google.com/github/sonder-art/ia_p26/blob/main/clase/15_adversarial_search/notebooks/02_minimax_y_alphabeta.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |

---

# Poda alfa-beta: la misma respuesta, menos trabajo

> *"The art of being wise is the art of knowing what to overlook."*
> — William James

---

Alpha-beta es minimax con una poda inteligente: descarta ramas del árbol que no pueden cambiar la decisión final. La respuesta es **idéntica** a minimax — la acción que retorna es la misma. El trabajo se puede reducir hasta a la raíz cuadrada del de minimax.

---

## 1. Intuición: el detective que para cuando sabe suficiente

Antes de cualquier fórmula, un ejemplo en lenguaje natural:

> Eres MAX. Ya exploraste la acción A y encontraste que te garantiza un valor de **+1**.
>
> Ahora exploras la acción B. Bajas al nodo MIN de esa rama. MIN examina sus opciones y encuentra una que le da un valor de **−1** — MIN puede forzar que tú obtengas ≤ −1 por esa rama.
>
> ¿Necesitas seguir explorando el resto de la rama B?
>
> **No.** MIN puede forzar ≤ −1 en esa rama, y tú ya tienes **+1** garantizado en la acción A. Nunca elegirías B. Todo lo que quede por explorar en esa rama no puede mejorar tu situación — puedes ignorarlo.

Ese razonamiento es alpha-beta. La poda ocurre cuando sabemos que, sin importar lo que quede por explorar en una rama, la decisión en el nodo ancestro no puede cambiar.

---

## 2. Los valores $\alpha$ y $\beta$

- $\alpha$ = el **mejor valor** (el mayor) que MAX puede **garantizarse** en el camino actual desde la raíz hasta el nodo actual. Empieza en $-\infty$.
- $\beta$ = el **mejor valor** (el menor) que MIN puede **garantizarse** en el camino actual desde la raíz hasta el nodo actual. Empieza en $+\infty$.

Se pasan como parámetros **hacia abajo** en el árbol — cada nodo hijo hereda los valores de su padre. La invariante es:

$$\alpha \leq V(s) \leq \beta$$

Si en algún momento $\alpha \geq \beta$, sabemos que este nodo no puede influir en la decisión del ancestro — podamos todo lo que quede por explorar en este subárbol.

---

## 3. Condición de poda

La poda ocurre cuando la **ventana** $[\alpha, \beta]$ se cierra, es decir, cuando el valor que el nodo actual puede forzar ya queda fuera del rango que le interesa al ancestro.

**Poda en nodo MAX** (β-cutoff): MAX encontró un hijo con valor $v \geq \beta$

```
Situación:
  - β es el mejor resultado que MIN puede garantizarse en otra rama.
  - MAX ahora puede forzar ≥ v ≥ β en esta rama.
  - MIN nunca elegiría venir a este subárbol porque ya tiene algo mejor (≤ β) en otro lado.
  → Podamos: no importa lo que quede por explorar aquí.

Condición: v ≥ β  →  PODA (β-cutoff)  →  return v inmediatamente
```

**Poda en nodo MIN** (α-cutoff): MIN encontró un hijo con valor $v \leq \alpha$

```
Situación:
  - α es el mejor resultado que MAX puede garantizarse en otra rama.
  - MIN ahora puede forzar ≤ v ≤ α en esta rama.
  - MAX nunca elegiría venir a este subárbol porque ya tiene algo mejor (≥ α) en otro lado.
  → Podamos: no importa lo que quede por explorar aquí.

Condición: v ≤ α  →  PODA (α-cutoff)  →  return v inmediatamente
```

Regla nemotécnica: **MAX poda cuando encontró algo tan bueno que MIN jamás lo permitiría** ($v \geq \beta$); **MIN poda cuando encontró algo tan malo para MAX que MAX jamás lo elegiría** ($v \leq \alpha$).

---

## 4. En lenguaje natural

El algoritmo en cuatro pasos — idéntico a minimax excepto por los parámetros $\alpha$ y $\beta$:

1. **Si el estado es terminal**: devuelve su utilidad. Igual que minimax.
2. **Si es el turno de MAX**: evalúa sucesores, actualiza $\alpha$ con el mejor encontrado. Si en cualquier momento $v \geq \beta$, **para** — el resto de esta rama no importa.
3. **Si es el turno de MIN**: evalúa sucesores, actualiza $\beta$ con el peor encontrado. Si en cualquier momento $v \leq \alpha$, **para** — el resto de esta rama no importa.
4. **En la raíz**: la decisión es idéntica a la de minimax. La diferencia es solo cuántos nodos se exploraron.

Los valores $\alpha$ y $\beta$ se pasan hacia abajo en cada llamada recursiva, acumulando la información de lo que ya se ha explorado en otras ramas del árbol.

---

## 5. Pseudocódigo

La estructura es **idéntica** a minimax (mismas tres funciones: `ALPHA_BETA`, `MAX_VALUE_AB`, `MIN_VALUE_AB`). La única diferencia es que ahora cada llamada recibe y actualiza dos parámetros extra: $\alpha$ y $\beta$.

Los cambios respecto a minimax son exactamente **tres líneas** — marcadas con `[P1]`, `[P2]`, `[P3]`:
- `[P1]` — se pasan $\alpha$ y $\beta$ en la llamada inicial (antes no existían estos parámetros).
- `[P2]` — dentro del bucle se actualiza $\alpha$ o $\beta$ con el mejor valor visto hasta ahora.
- `[P3]` — si la ventana $[\alpha, \beta]$ se cerró, se retorna antes de terminar el bucle (la poda).

```
# ── ALPHA_BETA ───────────────────────────────────────────────────────────────
# Punto de entrada. Igual que MINIMAX, pero pasa α y β hacia abajo.
# α y β empiezan en −∞ y +∞: todavía no sabemos nada del árbol.

function ALPHA_BETA(juego, estado):

    if juego.jugador(estado) == MAX:
        v, accion ← MAX_VALUE_AB(juego, estado, -∞, +∞)  # [P1] valores iniciales: sin información aún
    else:
        v, accion ← MIN_VALUE_AB(juego, estado, -∞, +∞)  # [P1] ídem para MIN
    return accion                    # devolvemos la ACCIÓN, igual que minimax


# ── MAX_VALUE_AB ─────────────────────────────────────────────────────────────
# Igual que MAX_VALUE en minimax, pero con dos diferencias:
#   - Actualiza α después de cada hijo explorado.
#   - Para antes de ver todos los hijos si ya no tiene sentido seguir (v ≥ β).

function MAX_VALUE_AB(juego, estado, α, β):

    if juego.terminal(estado):           # ¿El juego terminó aquí?
        return juego.utilidad(estado), None   # hoja: valor conocido, sin acción

    v ← -∞                              # peor valor posible para MAX
    mejor ← None                        # todavía no hay mejor acción
    for each accion in juego.acciones(estado):       # recorre cada movimiento legal
        sucesor ← juego.resultado(estado, accion)    # aplica el movimiento → estado hijo
        v2, _ ← MIN_VALUE_AB(juego, sucesor, α, β)  # evalúa ese hijo en el nivel MIN
        if v2 > v:                       # ¿es este hijo mejor que el mejor visto hasta ahora?
            v ← v2                       # sí: actualiza el mejor valor
            mejor ← accion              # guarda la acción que lo produjo
        α ← max(α, v)                   # [P2] actualiza α: MAX ahora garantiza al menos v
        if v ≥ β:                        # [P3] poda β: MAX puede forzar v ≥ β,
            return v, mejor             #      pero MIN ya tiene una opción ≤ β en otra rama
                                        #      → MIN nunca elegiría venir aquí → podamos
    return v, mejor                      # propaga el mejor valor encontrado


# ── MIN_VALUE_AB ─────────────────────────────────────────────────────────────
# Igual que MIN_VALUE en minimax, pero actualiza β y poda si v ≤ α.

function MIN_VALUE_AB(juego, estado, α, β):

    if juego.terminal(estado):           # ¿El juego terminó aquí?
        return juego.utilidad(estado), None   # hoja: valor conocido, sin acción

    v ← +∞                              # peor valor posible para MIN (el más alto)
    mejor ← None
    for each accion in juego.acciones(estado):       # recorre cada movimiento legal
        sucesor ← juego.resultado(estado, accion)    # aplica el movimiento → estado hijo
        v2, _ ← MAX_VALUE_AB(juego, sucesor, α, β)  # evalúa ese hijo en el nivel MAX
        if v2 < v:                       # ¿es este hijo peor para MAX (= mejor para MIN)?
            v ← v2                       # sí: actualiza el mejor valor para MIN
            mejor ← accion
        β ← min(β, v)                   # [P2] actualiza β: MIN ahora garantiza a lo sumo v
        if v ≤ α:                        # [P3] poda α: MIN puede forzar v ≤ α,
            return v, mejor             #      pero MAX ya tiene una opción ≥ α en otra rama
                                        #      → MAX nunca elegiría venir aquí → podamos
    return v, mejor
```

**Resumen de los tres cambios sobre minimax:**

| Cambio | Dónde | Qué hace |
|---|---|---|
| `[P1]` | Punto de entrada | Pasa $\alpha=-\infty$, $\beta=+\infty$ — sin información inicial |
| `[P2]` | Dentro del bucle | Actualiza $\alpha$ (MAX) o $\beta$ (MIN) con el mejor valor encontrado hasta ahora |
| `[P3]` | Dentro del bucle | Si la ventana $[\alpha, \beta]$ se cierra, para — el resto de la rama no puede cambiar la decisión |

---

## 6. Implementación en Python

```python
def alpha_beta(estado, es_max, alpha=-float('inf'), beta=float('inf')):
    """
    Alpha-beta para Nim.

    Args:
        estado : tupla de enteros — tamaño de cada pila, e.g. (2, 3)
        es_max : True si es turno de MAX, False si es turno de MIN
        alpha  : mejor valor que MAX puede garantizarse en el camino desde la raíz hasta aquí
        beta   : mejor valor que MIN puede garantizarse en el camino desde la raíz hasta aquí

    Returns:
        (valor, accion)  →  valor ∈ {+1, −1},  accion = (indice_pila, cantidad) o None
    """
    # ── Caso base: estado terminal ───────────────────────────────────────────
    # Todas las pilas vacías: el jugador en turno no puede mover → pierde
    if all(p == 0 for p in estado):
        return (-1 if es_max else +1), None   # MAX pierde si es su turno, MIN pierde si es el suyo

    # ── Inicialización ───────────────────────────────────────────────────────
    # MAX empieza queriendo el mayor valor posible → inicia en −∞
    # MIN empieza queriendo el menor valor posible → inicia en +∞
    mejor_valor = -float('inf') if es_max else float('inf')
    mejor_accion = None

    # ── Exploración de sucesores ─────────────────────────────────────────────
    for i, pila in enumerate(estado):           # i = índice de la pila
        for cant in range(1, pila + 1):         # cant ∈ {1, …, tamaño_pila}
            nuevo = list(estado)
            nuevo[i] -= cant                    # aplica el movimiento: retira 'cant' de pila i
            # Llamada recursiva: el turno alterna, α y β bajan iguales
            v, _ = alpha_beta(tuple(nuevo), not es_max, alpha, beta)

            if es_max:
                # ── Nodo MAX ─────────────────────────────────────────────────
                if v > mejor_valor:             # ¿este hijo es mejor para MAX?
                    mejor_valor = v
                    mejor_accion = (i, cant)
                alpha = max(alpha, mejor_valor) # MAX ahora garantiza al menos mejor_valor
                if mejor_valor >= beta:         # poda β: MAX puede forzar ≥ β,
                    return mejor_valor, mejor_accion   # pero MIN nunca vendría aquí
            else:
                # ── Nodo MIN ─────────────────────────────────────────────────
                if v < mejor_valor:             # ¿este hijo es peor para MAX (= mejor para MIN)?
                    mejor_valor = v
                    mejor_accion = (i, cant)
                beta = min(beta, mejor_valor)   # MIN ahora garantiza a lo sumo mejor_valor
                if mejor_valor <= alpha:        # poda α: MIN puede forzar ≤ α,
                    return mejor_valor, mejor_accion   # pero MAX nunca vendría aquí

    return mejor_valor, mejor_accion


# ── Verificación: alpha-beta == minimax ──────────────────────────────────────
# La acción elegida debe ser idéntica — solo cambia cuántos nodos se visitaron
if __name__ == '__main__':
    estado = (1, 2)
    v_mm, a_mm = minimax(estado, es_max=True)
    v_ab, a_ab = alpha_beta(estado, es_max=True)
    assert v_mm == v_ab, "Los valores deben coincidir"
    assert a_mm == a_ab, "Las acciones deben coincidir"
    print(f"Minimax:    valor={v_mm}, accion={a_mm}")
    print(f"Alpha-beta: valor={v_ab}, accion={a_ab}")
    # → Minimax:    valor=1, accion=(1, 1)
    # → Alpha-beta: valor=1, accion=(1, 1)
```

---

## 7. Traza: Nim(2,3) con poda

Para mostrar la poda de forma más visible usamos **Nim(2,3)**: pila A = 2 fichas, pila B = 3 fichas. El árbol es más grande que Nim(1,2) — hay más ramas desde la raíz — y por eso alpha-beta tiene más oportunidades de podar.

La decisión final es **idéntica** a minimax. La diferencia es que alpha-beta abandona varias ramas antes de terminar de explorarlas.

### Lectura de la figura

![Alpha-beta en Nim(2,3)]({{ '/15_adversarial_search/images/08_alphabeta_nim23.png' | url }})

**Cómo leer esta figura:**

- **Nodos azules**: turno de MAX (maximiza). **Nodos rojos**: turno de MIN (minimiza). Los colores alternan nivel a nivel, igual que en Nim(1,2).
- **Cada arista** es un movimiento legal, etiquetada con la acción (p.ej. `A-1` = retirar 1 de pila A, `B-2` = retirar 2 de pila B).
- **El par ($\alpha$, $\beta$)** anotado junto a cada nodo muestra los valores en el momento en que ese nodo se evalúa. $\alpha$ es el mejor que MAX ya sabe garantizarse en el camino desde la raíz; $\beta$ es el mejor que MIN ya sabe garantizarse. Ambos valores bajan del padre al hijo — la información se *hereda hacia abajo*.
- **Ramas en gris con ✗**: ramas podadas. Esos nodos nunca se expanden. Alpha-beta decidió que explorarlos no puede cambiar la decisión en el ancestro.
- **Nodos terminales** (cuadrados): estado (0,0). Valor +1 si le toca a MIN en ese terminal (MIN pierde), −1 si le toca a MAX (MAX pierde). Igual que en Nim(1,2).
- **Nodo raíz (2,3)**: su valor final (+1) es el mismo que obtendría minimax. MAX elige la misma acción óptima — con menos trabajo.

**Qué significa una poda concretamente**: cuando una rama se poda, significa que aunque exploráramos todos sus hijos, el resultado no podría mejorar la decisión del ancestro. Alguien en el camino hacia la raíz ya tiene una opción mejor garantizada — ese "alguien" nunca elegiría esta rama.

### Traza parcial — el momento de la primera poda

| # | Estado | Jugador | $\alpha$ | $\beta$ | Retorna | Nota |
|:--:|:---:|:---:|:---:|:---:|---|---|
| 1 | (2,3) | MAX | $-\infty$ | $+\infty$ | — | inicio |
| 2 | (1,3) | MIN | $-\infty$ | $+\infty$ | — | acción A-1 |
| 3 | (0,3) | MAX | $-\infty$ | $+\infty$ | — | MIN→A-1 |
| 4 | (0,2) | MIN | $-\infty$ | $+\infty$ | — | MAX→B-1 |
| 5 | (0,1) | MAX | $-\infty$ | $+\infty$ | — | MIN→B-1 |
| 6 | (0,0) | MAX | $-\infty$ | $+\infty$ | **−1** ✓ | terminal |
| 7 | ← (0,1) | MAX | $-\infty$ | $+\infty$ | **+1** | única acción |
| 8 | ← (0,2) | MIN | $-\infty$ | **+1** | — | β ← +1 |
| 9 | (0,0) | MIN | $-\infty$ | **+1** | **+1** ✓ | terminal |
| 10 | ← (0,2) | MIN | $-\infty$ | **0** | — | min(+1,+1) sigue |
| 11 | ← (0,3) | MAX | **+1** | $+\infty$ | **+1** | α ← +1 |
| 12 | (1,2) | MIN | **+1** | $+\infty$ | — | MIN acción A-1 (pila A=1) |
| 13 | (0,2) | MAX | **+1** | $+\infty$ | — | MAX evalúa |
| 14 | (0,1) | MIN | **+1** | $+\infty$ | — | |
| 15 | (0,0) | MIN | **+1** | $+\infty$ | **+1** ✓ | terminal |
| 16 | ← (0,1) | MIN | **+1** | **+1** | **+1** | β ← +1; **+1 ≤ α=+1 → PODA** |
| 17 | (1,2) poda | MIN | **+1** | **+1** | **≤+1** | ramas restantes de (1,2) podadas |
| 18 | ← (1,3) | MIN | **+1** | $+\infty$ | min(+1, ≤+1) = | MIN no puede superar lo ya visto |
| 19 | ← (2,3) | MAX | **+1** | $+\infty$ | max(+1,…) = **+1** | acción A-1 confirmada óptima |

![Minimax vs alpha-beta]({{ '/15_adversarial_search/images/09_alphabeta_vs_minimax.png' | url }})

La figura compara el número de nodos expandidos por minimax y alpha-beta en el mismo árbol de Nim(2,3). Cada barra representa un juego distinto — el eje horizontal es el juego, el eje vertical es cuántos nodos se visitaron. La barra azul es minimax (visita todos los nodos); la barra naranja es alpha-beta (visita un subconjunto). En ambos casos, la **acción elegida es idéntica** — la reducción es solo en trabajo, no en calidad.

---

## 8. Análisis de eficiencia

| Ordenamiento | Complejidad | Equivalencia | Ejemplo: ajedrez a profundidad 4 |
|---|---|---|---|
| **Peor caso** | $O(b^m)$ | = minimax | Sin ahorro — poda nunca dispara |
| **Orden aleatorio** | $O(b^{3m/4})$ | ≈ 25% menos profundidad | Equivale a explorar hasta profundidad 3 |
| **Orden perfecto** | **$O(b^{m/2})$** | **dobla la profundidad efectiva** | **→ profundidad 8 con mismo cómputo** |

Con ordenamiento perfecto, alpha-beta puede buscar el **doble de profundidad** que minimax con el mismo tiempo de cómputo. Para ajedrez: pasar de profundidad 4 a profundidad 8 es una diferencia enorme en calidad de juego — la diferencia entre un programa que comete errores tácticos obvios y uno que planea combinaciones de varias jugadas.

La intuición del ordenamiento perfecto: si siempre exploramos primero el mejor movimiento (desde la perspectiva de quien elige), el valor de $\alpha$ sube (o $\beta$ baja) tan rápido como es posible, generando cortes en todos los hermanos restantes.

---

## 9. Ordenamiento de movimientos y módulo 14

El ordenamiento de movimientos es el análogo adversarial de las heurísticas $h(n)$ del módulo 14:

> En A\*, ordenar la frontera por $h(n)$ guía la búsqueda hacia la meta más rápidamente. En alpha-beta, explorar primero los movimientos más prometedores genera cortes $\alpha$/$\beta$ más pronto, reduciendo el trabajo total.

Estrategias comunes de ordenamiento:
- **Killer moves**: movimientos que causaron cortes en otras ramas del mismo nivel — es probable que también causen cortes aquí.
- **Evaluación rápida**: calcular un valor aproximado de cada movimiento antes de ordenarlos. Una función análoga a $h$ que mide el "atractivo" del movimiento.
- **Tabla de transposición**: recordar qué posiciones ya se evaluaron y su valor — si llegamos a la misma posición por otra secuencia, reutilizamos el resultado.

Todas estas técnicas hacen que alpha-beta se acerque al caso de ordenamiento perfecto — y nos llevan naturalmente a las funciones de evaluación de la sección 15.5.

---

## 10. Una posición real de tic-tac-toe

![Tic-tac-toe: decisión minimax]({{ '/15_adversarial_search/images/10_tictactoe_endgame.png' | url }})

La figura muestra una posición de tic-tac-toe donde O amenaza ganar en la siguiente jugada (tiene dos en raya con una celda libre). Minimax identifica el único movimiento correcto de X — bloquear la amenaza — y calcula que el juego termina en empate con juego perfecto de ambos lados. Alpha-beta llega a la misma conclusión expandiendo un subconjunto estrictamente menor de nodos: en cuanto encuentra el bloqueo obligatorio, poda todas las ramas alternativas.

---

## 11. ¿Y el ajedrez?

Alpha-beta con ordenamiento perfecto puede alcanzar profundidad 8 en lugar de profundidad 4 con el mismo cómputo. Eso es la diferencia entre un motor "principiante" y un motor de "jugador de club". Pero aun así, profundidad 8 con $b \approx 35$ significa $\approx 35^4 \approx 1.5 \times 10^6$ nodos — manejable. Profundidad 12, que es lo que logran los motores modernos, requiere técnicas adicionales que veremos en la siguiente sección.

---

**Siguiente:** [Juegos complejos →](05_juegos_complejos.md)
