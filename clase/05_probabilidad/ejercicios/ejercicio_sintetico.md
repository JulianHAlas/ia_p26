# Ejercicio: Anatomía de las Colas Largas (Sintético)

## Objetivo

Experimentar con distribuciones fat-tailed en un entorno controlado para entender visceralmente:
- Por qué el promedio converge (o no)
- El efecto devastador de una sola observación extrema
- El criterio κ de Taleb
- Por qué Mediocristán ≠ Extremistán

## Contexto Teórico

### Los Tres Regímenes

Para distribuciones de ley de potencias con exponente α:

| Régimen | Media | Varianza | LGN | TLC | Comportamiento |
|---------|-------|----------|-----|-----|----------------|
| α > 2 | ✓ Finita | ✓ Finita | ✓ Funciona | ✓ Funciona | "Normal" |
| 1 < α ≤ 2 | ✓ Finita | ✗ Infinita | ⚠️ Muy lento | ✗ Falla | Fat-tailed |
| α ≤ 1 | ✗ Infinita | ✗ Infinita | ✗ Falla | ✗ Falla | Extremo |

### Distribución de Pareto

$$P(X > x) = \left(\frac{x_m}{x}\right)^\alpha \quad \text{para } x \geq x_m$$

- Si α = 3: media = 1.5, varianza finita
- Si α = 2: media = 2, varianza infinita
- Si α = 1.5: media = 3, varianza infinita
- Si α = 1: media infinita (Zipf)

### Distribución de Cauchy

Caso extremo donde **ni la media existe**. El promedio de n variables Cauchy es... Cauchy.

---

## Lo Que Harás

### Parte A: El Experimento de Convergencia

**Pregunta:** ¿Cuántas observaciones necesitas para que el promedio se estabilice?

1. Simular 30 trayectorias del promedio acumulado para:
   - Normal(0, 1)
   - Pareto(α=3)
   - Pareto(α=2)
   - Pareto(α=1.5)

2. Para cada una, medir:
   - ¿A qué n el 90% de trayectorias están dentro del ±10% de la media teórica?
   - ¿Qué tan "suaves" vs "erráticas" son las trayectorias?

3. Llenar la tabla:

| Distribución | n para convergencia 10% |
|--------------|-------------------------|
| Normal | ~? |
| Pareto α=3 | ~? |
| Pareto α=2 | ~? |
| Pareto α=1.5 | ~? |

### Parte B: El Experimento "Un Cisne Negro"

**Pregunta:** ¿Cuánto puede cambiar el promedio al añadir UNA observación?

1. Generar 1000 observaciones de cada distribución
2. Calcular: media, mediana, desviación estándar
3. Añadir UNA observación del cuantil 99.9%
4. Recalcular las estadísticas
5. Medir el cambio porcentual

| Distribución | Cambio en media | Cambio en mediana | Cambio en σ |
|--------------|-----------------|-------------------|-------------|
| Normal | ?% | ?% | ?% |
| Pareto α=1.5 | ?% | ?% | ?% |

**Lección esperada:** La mediana es robusta, la media no.

### Parte C: El Criterio κ de Taleb

**Pregunta:** ¿Qué fracción del total aporta la observación más grande?

$$\kappa = \frac{\max(X_1, \ldots, X_n)}{\sum_{i=1}^n X_i}$$

1. Calcular κ para muestras de tamaño n = 100, 500, 1000, 5000
2. Repetir 500 veces para cada n
3. Graficar la distribución de κ

**Interpretación:**
- κ → 0: ninguna observación domina (thin-tailed)
- κ permanece alto: "winner takes all" (fat-tailed)

### Parte D: Curva de Concentración (Lorenz)

**Pregunta:** ¿Qué porcentaje del total aporta el top X%?

1. Generar 10,000 observaciones
2. Ordenar de mayor a menor
3. Calcular: ¿cuánto aporta el top 1%? ¿Top 10%? ¿Top 20%?

| Distribución | Top 1% | Top 10% | Top 20% |
|--------------|--------|---------|---------|
| Normal | ?% | ?% | ?% |
| Pareto α=2 | ?% | ?% | ?% |
| Pareto α=1.5 | ?% | ?% | ?% |

**Lección esperada:** En fat tails, "el 20% tiene el 80%" (o peor).

---

## Preguntas de Reflexión

1. **¿Por qué la mediana es más robusta que la media?** ¿En qué situaciones preferirías usar mediana?

2. **Si κ permanece alto sin importar n, ¿qué implica para el muestreo?** ¿Puedes "promediar" el riesgo away?

3. **Piensa en la distribución de riqueza.** Si la riqueza es Pareto con α ≈ 1.5, ¿qué porcentaje del total tiene el 1% más rico?

4. **Piensa en el éxito de productos/apps.** Si las descargas siguen ley de potencias, ¿tiene sentido hablar del "app promedio"?

5. **¿Por qué es peligroso usar la media cuando sospechas fat tails?** Da un ejemplo concreto.

---

## Extensiones Sugeridas

1. **Explorar otros valores de α:** ¿Qué pasa en la "frontera" α = 2?

2. **Simular Cauchy:** Ver que el promedio de Cauchy es Cauchy (no converge nunca)

3. **Mezclas de distribuciones:** ¿Qué pasa si el 99% de los datos son normales pero el 1% son fat-tailed?

4. **Comparar con datos reales:** ¿Los retornos del S&P 500 se parecen más a Normal o a Pareto?

---

## Ejecutar

```bash
cd clase/05_probabilidad/ejercicios
python ejercicio_sintetico.py
```

El script generará:
- Gráficas de convergencia para cada distribución
- Análisis del impacto de un cisne negro
- Histogramas de κ
- Curvas de Lorenz
- Todas guardadas en `outputs/`

---

## Nota

Este ejercicio complementa el laboratorio principal (`../lab_probabilidad.py`) que genera las imágenes para las notas de clase. Aquí el enfoque es más interactivo: tú modificas parámetros y observas los efectos.
