# Ejercicio: Los Eventos Imposibles del S&P 500

## Objetivo

Demostrar que los retornos financieros **NO** siguen una distribución normal, usando datos históricos reales del S&P 500.

## Contexto Teórico

### La Hipótesis de Mercados Eficientes y Normalidad

Durante décadas, la teoría financiera asumió que los retornos de acciones siguen una distribución normal (o log-normal). Esta suposición es conveniente porque:

1. La normal está completamente caracterizada por μ y σ
2. El TLC sugiere que sumas de muchos factores → normal
3. Permite fórmulas cerradas (Black-Scholes, VaR, etc.)

### El Problema

Si los retornos fueran normales:

| Evento | Probabilidad | Frecuencia esperada |
|--------|--------------|---------------------|
| > 3σ | 0.27% | 1 vez cada ~370 días (~1.5 años) |
| > 4σ | 0.006% | 1 vez cada ~15,800 días (~63 años) |
| > 5σ | 0.00006% | 1 vez cada ~1.7 millones de días (~6,900 años) |
| > 6σ | 2×10⁻⁷% | 1 vez cada ~500 millones de días (~1.4 millones de años) |

Un evento de 6σ debería ocurrir aproximadamente **una vez desde que existían los dinosaurios**.

### La Realidad

El Lunes Negro (19 de octubre de 1987), el S&P 500 cayó **22.6%** en un solo día. Con la volatilidad histórica de ~1% diario, esto es un evento de aproximadamente **20-25 sigmas**.

La probabilidad de un evento de 20σ en una distribución normal es aproximadamente $10^{-88}$. Para contexto, hay aproximadamente $10^{80}$ átomos en el universo observable.

**O los modelos están mal, o presenciamos un milagro estadístico.**

---

## Lo Que Harás

### Parte A: Descargar y Preparar Datos

1. Descargar retornos diarios del S&P 500 desde 1950
2. Calcular retornos logarítmicos: $r_t = \ln(P_t / P_{t-1})$
3. Calcular media (μ) y desviación estándar (σ) de los retornos

### Parte B: Contar Eventos Extremos

4. Para cada umbral (3σ, 4σ, 5σ, 6σ), contar:
   - Cuántos días el retorno fue más extremo (positivo o negativo)
   - Comparar con lo esperado bajo normalidad

5. Crear una tabla comparativa:

```
| Umbral | Normal predice | Realidad | Factor de subestimación |
|--------|----------------|----------|-------------------------|
| > 3σ   | X días         | Y días   | Y/X                     |
| > 4σ   | ...            | ...      | ...                     |
```

### Parte C: Identificar los Cisnes Negros

6. Listar las fechas de los 20 eventos más extremos
7. Para cada uno, calcular cuántos sigmas representa
8. Investigar (Google) qué pasó esos días

### Parte D: Visualizar

9. Histograma de retornos vs distribución normal teórica
10. QQ-plot para ver la desviación en las colas
11. Gráfica de eventos extremos a lo largo del tiempo

---

## Preguntas de Reflexión

Después de ejecutar el código y ver los resultados, reflexiona:

1. **¿Cuántos eventos "imposibles" encontraste?** ¿Cuántos de ellos ocurrieron en tu vida?

2. **Si los modelos normales están tan mal, ¿por qué se siguen usando?** Piensa en incentivos, conveniencia, y "todos los demás lo hacen".

3. **¿Qué implicaciones tiene esto para tu estrategia de inversión personal?** ¿Confiarías en un asesor que usa modelos normales?

4. **Investiga el Lunes Negro (1987).** ¿Qué lo causó? ¿Había señales previas? ¿Podría pasar de nuevo?

5. **¿Por qué crees que los eventos extremos tienden a agruparse?** (Nota: no son independientes)

---

## Extensiones Sugeridas

Si quieres ir más allá:

1. **Otros activos:** Repite el análisis con Bitcoin (`BTC-USD`), oro (`GC=F`), o acciones individuales (AAPL, TSLA)

2. **Diferentes períodos:** ¿Los retornos se han vuelto más o menos extremos con el tiempo?

3. **Ajuste de distribución:** Intenta ajustar una Student-t a los datos. ¿Qué grados de libertad dan mejor fit?

4. **Correlación en las colas:** ¿Los eventos extremos en diferentes activos tienden a ocurrir juntos?

---

## Ejecutar

```bash
cd clase/05_probabilidad/ejercicios
python ejercicio_sp500.py
```

El script generará:
- Tabla de eventos esperados vs observados
- Lista de los eventos más extremos con fechas
- Gráficas guardadas en `outputs/`
