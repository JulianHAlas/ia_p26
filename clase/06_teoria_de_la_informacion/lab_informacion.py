#!/usr/bin/env python3
"""
Laboratorio: Teoría de la Información (imágenes + experimentos)

Uso:
    cd clase/06_teoria_de_la_informacion
    python lab_informacion.py

Genera imágenes en:
    clase/06_teoria_de_la_informacion/images/

Este laboratorio está diseñado para enseñar en clase:
- bits como preguntas
- auto-información y entropía
- cross-entropy y KL
- ganancia esperada de información en Wordle (a escala pequeña)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from it_code.info_math import entropy_bits, cross_entropy_bits, kl_divergence_bits
from it_code.lexicons import load_generated_spanish_5letter, load_mini_spanish_5letter
from it_code.wordle import feedback_pattern


# -----------------------------------------------------------------------------
# Styling (similar vibe to lab_probabilidad.py)
# -----------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

COLORS = {
    "blue": "#2E86AB",
    "red": "#E94F37",
    "green": "#27AE60",
    "gray": "#7F8C8D",
}


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

np.random.seed(42)


def _save(fig, name: str) -> None:
    out = IMAGES_DIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Generada: {out.name}")


# -----------------------------------------------------------------------------
# 1) Entropy vs concentration (simple UX)
# -----------------------------------------------------------------------------


def plot_entropy_two_outcomes():
    ps = np.linspace(1e-6, 1 - 1e-6, 800)
    hs = [entropy_bits([p, 1 - p]) for p in ps]

    fig, ax = plt.subplots()
    ax.plot(ps, hs, color=COLORS["blue"], linewidth=2)
    ax.set_title("Entropía de Bernoulli: H(p) (bits)")
    ax.set_xlabel("p")
    ax.set_ylabel("H(p)")
    ax.axvline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.annotate("Máximo en p=0.5", xy=(0.5, 1.0), xytext=(0.62, 0.85),
                arrowprops=dict(arrowstyle="->"))
    _save(fig, "entropia_bernoulli.png")


def plot_entropy_dirichlet_like():
    """
    Show entropy as distribution becomes more concentrated.
    We build a family of distributions over N symbols by interpolating
    between uniform and a 'peaked' distribution.
    """
    N = 12
    uniform = np.ones(N) / N
    peaked = np.zeros(N)
    peaked[0] = 1.0

    alphas = np.linspace(0, 1, 101)
    hs = []
    for a in alphas:
        p = (1 - a) * uniform + a * peaked
        hs.append(entropy_bits(p))

    fig, ax = plt.subplots()
    ax.plot(alphas, hs, color=COLORS["red"], linewidth=2)
    ax.set_title("Entropía disminuye cuando el prior se concentra")
    ax.set_xlabel("a (0=uniforme, 1=concentrado)")
    ax.set_ylabel("H (bits)")
    ax.annotate("Uniforme", xy=(0, hs[0]), xytext=(0.05, hs[0] - 0.5))
    ax.annotate("Concentrado", xy=(1, hs[-1]), xytext=(0.65, hs[-1] + 0.4))
    _save(fig, "entropia_concentracion.png")


# -----------------------------------------------------------------------------
# 2) Cross-entropy & KL demo
# -----------------------------------------------------------------------------


def plot_cross_entropy_vs_model_mismatch():
    """
    Fix true p over 3 classes and vary q along a path to show:
    H(p,q) and D_KL(p||q) track mismatch.
    """
    p = np.array([0.7, 0.2, 0.1])
    q_good = np.array([0.6, 0.25, 0.15])
    q_bad = np.array([0.4, 0.3, 0.3])

    ts = np.linspace(0, 1, 201)
    h_p = entropy_bits(p)
    hs = []
    kls = []
    for t in ts:
        q = (1 - t) * q_good + t * q_bad
        q = q / q.sum()
        hs.append(cross_entropy_bits(p, q))
        kls.append(kl_divergence_bits(p, q))

    fig, ax = plt.subplots()
    ax.plot(ts, hs, label="H(p,q) (cross-entropy)", color=COLORS["blue"], linewidth=2)
    ax.plot(ts, [h_p] * len(ts), label="H(p) (entropía verdadera)", color=COLORS["gray"], linestyle="--")
    ax.plot(ts, [h_p + d for d in kls], label="H(p)+D_KL(p||q)", color=COLORS["red"], alpha=0.8)
    ax.set_title("Cross-entropy = entropía + KL (bits)")
    ax.set_xlabel("t (0=q_good → 1=q_bad)")
    ax.set_ylabel("bits")
    ax.legend()
    _save(fig, "cross_entropy_kl_identidad.png")


# -----------------------------------------------------------------------------
# 3) Wordle expected information gain (small scale)
# -----------------------------------------------------------------------------


def _entropy_of_word_posterior(weights: Dict[str, float], words: List[str]) -> float:
    ws = np.array([weights[w] for w in words], dtype=float)
    ws = ws / ws.sum()
    return entropy_bits(ws)


def expected_entropy_after_guess(
    words: List[str],
    weights: Dict[str, float],
    guess: str,
) -> float:
    """
    Computes E_F[ H(X | F, I) ] for a given guess.

    Complexity: O(N^2) in worst case (fine for demo N<=500).
    """
    # Probability of each pattern under current prior
    pattern_mass: Dict[Tuple[int, int, int, int, int], float] = {}
    # Group secrets by pattern to compute posterior entropies efficiently
    pattern_words: Dict[Tuple[int, int, int, int, int], List[str]] = {}

    total = sum(weights[w] for w in words)
    for secret in words:
        pat = feedback_pattern(secret, guess)
        m = weights[secret] / total
        pattern_mass[pat] = pattern_mass.get(pat, 0.0) + m
        pattern_words.setdefault(pat, []).append(secret)

    exp_h = 0.0
    for pat, mass in pattern_mass.items():
        post_words = pattern_words[pat]
        # Posterior over remaining candidates is proportional to prior restricted to this set
        exp_h += mass * _entropy_of_word_posterior(weights, post_words)
    return exp_h


def plot_wordle_expected_information_gain(max_words: int = 200):
    lex = load_generated_spanish_5letter(ROOT) or load_mini_spanish_5letter(ROOT)

    # For speed/UX, restrict to top-N by weight
    words = sorted(lex.words, key=lambda w: lex.weights.get(w, 0.0), reverse=True)[:max_words]
    weights = {w: max(lex.weights.get(w, 1.0), 1e-12) for w in words}

    base_h = _entropy_of_word_posterior(weights, words)

    # Candidate guesses: use same set (good enough for teaching)
    guesses = words[: min(80, len(words))]
    igs = []
    for g in guesses:
        exp_h = expected_entropy_after_guess(words, weights, g)
        igs.append((g, base_h - exp_h))

    igs.sort(key=lambda t: t[1], reverse=True)
    top = igs[:20]

    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [w for (w, _) in top][::-1]
    vals = [v for (_, v) in top][::-1]
    ax.barh(labels, vals, color=COLORS["green"], alpha=0.9)
    ax.set_title(f"Top guesses por ganancia esperada de información (N={len(words)})")
    ax.set_xlabel("IG(g) = H - E[H | feedback]  (bits)")
    ax.set_ylabel("guess")
    _save(fig, "wordle_top_info_gain.png")


def main() -> int:
    print("Generando imágenes (Teoría de la Información)...")
    plot_entropy_two_outcomes()
    plot_entropy_dirichlet_like()
    plot_cross_entropy_vs_model_mismatch()
    plot_wordle_expected_information_gain()
    print("Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

