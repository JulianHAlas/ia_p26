---
title: "Ejercicios: Wordle/Password con entropía (capstone)"
---

# Ejercicios: Wordle/Password con entropía (capstone)

Este capstone cierra el módulo con un ejercicio acumulativo:

- **Wordle/Fallout hacking** (con feedback): eliges un guess para maximizar reducción esperada de entropía.
- **Password guessing** (sin feedback): ordenas intentos usando priors (ranking) y comparas estrategias.

La idea es que veas la misma teoría aplicada en dos “regímenes”:

- con información rica por intento (Wordle),
- con información mínima por intento (password guessing).

## Setup (datasets opcionales)

Recomendado (para priors realistas):

```bash
cd clase/06_teoria_de_la_informacion
python -m datasets.download_datasets
python -m datasets.prepare_lexicons
```

Si no hay internet, el ejercicio funciona con un fallback pequeño (offline).

## Ejecutar el capstone

Desde `clase/06_teoria_de_la_informacion/`:

```bash
python ejercicios/capstone_wordle_password.py --mode wordle --random
python ejercicios/capstone_wordle_password.py --mode password
```

## Referencia (inspiración)

- [3Blue1Brown: Solving Wordle using information theory](https://www.3blue1brown.com/lessons/wordle)

## Qué debes observar (objetivos)

1. **Entropía inicial**: cuánta incertidumbre tienes al inicio (en bits).
2. **Ganancia esperada por guess** (Wordle): qué guesses “parten” mejor el espacio de candidatos.
3. **Priors importan**: el mejor guess cambia si tu distribución está sesgada.
4. **Sin feedback**: el problema password se vuelve “ordenar intentos” y medir el costo esperado.

## Enlace con el laboratorio (imágenes para explicar)

Si corriste:

```bash
python lab_informacion.py
```

Se generan imágenes como:

- `images/entropia_bernoulli.png` (entropía vs concentración)
- `images/cross_entropy_kl_identidad.png` (cross-entropy = entropía + KL)
- `images/wordle_top_info_gain.png` (top guesses por IG)

Estas se referencian en las notas del módulo.

---

**Volver:** [← Índice del módulo](../00_index.md)

