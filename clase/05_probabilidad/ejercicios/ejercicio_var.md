# Ejercicio: El Fraude del Value at Risk (VaR)

## Objetivo

Demostrar por qué el VaR (Value at Risk) —la medida de riesgo más usada en finanzas— falla sistemáticamente cuando los retornos tienen colas pesadas.

## Contexto Teórico

### ¿Qué es el VaR?

El Value at Risk responde la pregunta:

> "¿Cuál es la pérdida máxima que no superaremos en X% de los casos?"

**Definición formal:** VaR al nivel de confianza α es el cuantil (1-α) de la distribución de pérdidas.

$$\text{VaR}_\alpha = -Q_{1-\alpha}(R)$$

donde $Q_p$ es el cuantil p de los retornos R.

**Ejemplos:**
- VaR₉₉ con horizonte de 1 día: "El 99% de los días, no perderemos más que esta cantidad"
- VaR₉₅ con horizonte de 10 días: "El 95% de las veces en 10 días, no perderemos más que esta cantidad"

### Cómo se Calcula: Método Paramétrico

El método más común asume normalidad:

$$R \sim \mathcal{N}(\mu, \sigma^2)$$

Entonces:

$$\text{VaR}_\alpha = -(\mu + z_{1-\alpha} \cdot \sigma)$$

donde $z_{1-\alpha}$ es el cuantil de la normal estándar:
- VaR₉₉: $z_{0.01} = -2.326$
- VaR₉₅: $z_{0.05} = -1.645$

### El Problema

Si los retornos son fat-tailed (como en realidad son), el VaR paramétrico:

1. **Subestima la frecuencia** de violaciones
2. **Subestima la severidad** de las pérdidas cuando hay violación

### Backtesting del VaR

Para evaluar un modelo de VaR, hacemos "backtesting":

1. Calculamos VaR para cada día usando datos hasta ese día
2. Comparamos con el retorno real del día siguiente
3. Contamos "violaciones" (días donde la pérdida superó el VaR)

**Si el modelo es correcto:**
- VaR₉₉ debería violarse ~1% de los días
- VaR₉₅ debería violarse ~5% de los días

---

## Lo Que Harás

### Parte A: Calcular VaR Paramétrico

1. Usar una ventana móvil de N días para estimar μ y σ
2. Calcular VaR₉₉ y VaR₉₅ para cada día
3. Comparar con el retorno real del día siguiente

### Parte B: Backtest — Contar Violaciones

4. Contar días donde el retorno real < -VaR
5. Calcular tasa de violación: violaciones / total de días
6. Comparar con la tasa esperada (1% para VaR₉₉, 5% para VaR₉₅)

### Parte C: Severidad de las Violaciones

7. Cuando el VaR falla, ¿por cuánto falla?
8. Calcular el "Expected Shortfall" empírico:
   - Promedio de pérdidas en los días que violaron el VaR
9. Comparar con el VaR (ratio pérdida real / VaR predicho)

### Parte D: Comparar con Modelo Fat-Tailed

10. Recalcular VaR usando distribución Student-t
11. Ver si las tasas de violación mejoran
12. Encontrar los grados de libertad óptimos

---

## Métricas Clave

### Tasa de Violación (Coverage)

$$\text{Tasa de Violación} = \frac{\text{Número de violaciones}}{\text{Total de días}}$$

- VaR₉₉ correcto: tasa ≈ 1%
- VaR₉₅ correcto: tasa ≈ 5%

### Ratio de Severidad

$$\text{Ratio} = \frac{\text{Pérdida real cuando hay violación}}{\text{VaR predicho}}$$

Si el modelo es correcto, este ratio debería estar cerca de 1.
Si el modelo subestima, el ratio será >> 1.

### Expected Shortfall (ES)

$$\text{ES}_\alpha = E[\text{Pérdida} | \text{Pérdida} > \text{VaR}_\alpha]$$

"Cuando las cosas van mal, ¿qué tan mal van?"

---

## Preguntas de Reflexión

1. **¿Cuántas veces se viola el VaR₉₉ vs lo esperado?** ¿Qué tan grave es la subestimación?

2. **Cuando el VaR falla, ¿falla por poco o por mucho?** ¿Qué implica esto para la gestión de riesgo?

3. **¿Por qué los reguladores (Basilea) ahora prefieren Expected Shortfall sobre VaR?**

4. **¿Mejora el modelo Student-t?** ¿Qué grados de libertad dan mejor fit?

5. **Si fueras un regulador bancario, ¿confiarías en los reportes de VaR de los bancos?**

6. **¿Qué incentivos tienen los bancos para usar modelos que subestiman el riesgo?**

---

## La Conexión con 2008

En agosto de 2007, David Viniar (CFO de Goldman Sachs) dijo:

> "We were seeing things that were 25-standard deviation moves, several days in a row."

Un evento de 25σ tiene probabilidad $\approx 10^{-135}$ bajo normalidad.

**La explicación correcta:** No fueron eventos de 25σ. Los retornos simplemente no son normales. El modelo estaba mal, no la realidad.

---

## Extensiones Sugeridas

1. **Diferentes activos:** ¿El VaR falla igual para bonos, oro, Bitcoin?

2. **Diferentes horizontes:** ¿El problema es peor para VaR diario, semanal, o mensual?

3. **VaR condicional:** ¿Mejora si usamos modelos GARCH para la volatilidad?

4. **Pruebas estadísticas formales:** Implementar el test de Kupiec o Christoffersen

---

## Ejecutar

```bash
cd clase/05_probabilidad/ejercicios
python ejercicio_var.py
```

El script generará:
- Tabla de violaciones esperadas vs observadas
- Análisis de severidad de violaciones
- Comparación Normal vs Student-t
- Gráficas guardadas en `outputs/`
