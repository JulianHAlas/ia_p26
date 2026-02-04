#!/usr/bin/env python3
"""
Ejercicio: Anatomía de las Colas Largas (Sintético)

Experimentos controlados para entender el comportamiento de fat tails.

Ejecutar:
    python ejercicio_sintetico.py

Autor: Módulo de Probabilidad - IA P26
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Colores
COLORS = {
    'normal': '#2E86AB',
    'pareto3': '#27AE60',
    'pareto2': '#E67E22',
    'pareto15': '#E74C3C',
    'cauchy': '#9B59B6'
}

# Crear directorio de outputs
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Semilla para reproducibilidad
np.random.seed(42)


# =============================================================================
# PARTE A: EXPERIMENTO DE CONVERGENCIA
# =============================================================================

def experimento_convergencia(n_max=10000, n_runs=30):
    """
    Simula la convergencia del promedio para diferentes distribuciones.
    """
    print("="*60)
    print("PARTE A: Experimento de Convergencia")
    print("="*60)
    
    ns = np.arange(1, n_max + 1)
    
    # Definir distribuciones con sus medias teóricas
    distribuciones = [
        ('Normal(0,1)', lambda n: np.random.normal(0, 1, n), 0, COLORS['normal']),
        ('Pareto(α=3)', lambda n: stats.pareto.rvs(3, size=n), 1.5, COLORS['pareto3']),
        ('Pareto(α=2)', lambda n: stats.pareto.rvs(2, size=n), 2.0, COLORS['pareto2']),
        ('Pareto(α=1.5)', lambda n: stats.pareto.rvs(1.5, size=n), 3.0, COLORS['pareto15']),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    resultados = []
    
    for ax, (nombre, sampler, media_teorica, color) in zip(axes.flat, distribuciones):
        trayectorias = []
        
        # Simular múltiples trayectorias
        for _ in range(n_runs):
            data = sampler(n_max)
            running_mean = np.cumsum(data) / ns
            trayectorias.append(running_mean)
            ax.plot(ns, running_mean, alpha=0.3, linewidth=0.5, color=color)
        
        # Media teórica
        ax.axhline(y=media_teorica, color='black', linestyle='--', linewidth=2,
                  label=f'μ = {media_teorica}')
        
        # Banda de convergencia (±10%)
        if media_teorica != 0:
            banda = abs(media_teorica) * 0.1
            ax.axhspan(media_teorica - banda, media_teorica + banda, 
                      alpha=0.2, color='green', label='±10%')
        
        ax.set_xscale('log')
        ax.set_xlabel('n')
        ax.set_ylabel('Promedio acumulado')
        ax.set_title(nombre)
        ax.legend(loc='upper right')
        
        # Calcular n para convergencia
        trayectorias = np.array(trayectorias)
        if media_teorica != 0:
            dentro_banda = np.abs(trayectorias - media_teorica) < banda
            pct_convergido = dentro_banda.mean(axis=0)
            
            # Encontrar n donde 90% están dentro
            idx_conv = np.where(pct_convergido >= 0.9)[0]
            n_conv = ns[idx_conv[0]] if len(idx_conv) > 0 else ">10000"
        else:
            n_conv = "~100"  # Normal converge rápido
        
        resultados.append({'Distribución': nombre, 'n para 90% convergencia': n_conv})
        
        # Anotación
        if isinstance(n_conv, int) and n_conv < 10000:
            ax.annotate(f'90% converge\n~n={n_conv}', xy=(0.02, 0.98), 
                       xycoords='axes fraction', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Convergencia del Promedio ({n_runs} simulaciones, n hasta {n_max})', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sintetico_convergencia.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Guardado: {OUTPUT_DIR / 'sintetico_convergencia.png'}")
    
    # Mostrar tabla
    print("\n   Resultados de convergencia:")
    for r in resultados:
        print(f"   {r['Distribución']}: n ≈ {r['n para 90% convergencia']}")
    
    return resultados


# =============================================================================
# PARTE B: EXPERIMENTO "UN CISNE NEGRO"
# =============================================================================

def experimento_cisne_negro(n=1000, n_simulations=500):
    """
    Mide el impacto de añadir UNA observación extrema.
    """
    print("\n" + "="*60)
    print("PARTE B: Experimento 'Un Cisne Negro'")
    print("="*60)
    
    distribuciones = [
        ('Normal(0,1)', lambda: np.abs(np.random.normal(0, 1, n)), 
         lambda: np.abs(np.random.normal(0, 1, 1))[0]),
        ('Pareto(α=1.5)', lambda: stats.pareto.rvs(1.5, size=n),
         lambda: stats.pareto.rvs(1.5, size=1)[0]),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    resultados = []
    
    for ax, (nombre, gen_base, gen_extremo) in zip(axes, distribuciones):
        cambios_media = []
        cambios_mediana = []
        
        for _ in range(n_simulations):
            # Datos base
            data = gen_base()
            media_antes = data.mean()
            mediana_antes = np.median(data)
            
            # Añadir un extremo (cuantil 99.9%)
            extremo = np.percentile(data, 99.9) * 2  # Algo aún más extremo
            data_con_extremo = np.append(data, extremo)
            
            media_despues = data_con_extremo.mean()
            mediana_despues = np.median(data_con_extremo)
            
            # Cambio porcentual
            cambio_media = abs(media_despues - media_antes) / media_antes * 100
            cambio_mediana = abs(mediana_despues - mediana_antes) / mediana_antes * 100
            
            cambios_media.append(cambio_media)
            cambios_mediana.append(cambio_mediana)
        
        # Histograma de cambios en media
        ax.hist(cambios_media, bins=50, alpha=0.7, color='coral', 
               edgecolor='white', label='Cambio en Media')
        ax.hist(cambios_mediana, bins=50, alpha=0.7, color='steelblue', 
               edgecolor='white', label='Cambio en Mediana')
        
        ax.axvline(x=np.median(cambios_media), color='red', linestyle='--', 
                  linewidth=2, label=f'Mediana Δmedia: {np.median(cambios_media):.1f}%')
        
        ax.set_xlabel('Cambio porcentual (%)')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'{nombre}\nEfecto de añadir 1 observación extrema')
        ax.legend()
        
        resultados.append({
            'Distribución': nombre,
            'Cambio medio en Media': f'{np.mean(cambios_media):.1f}%',
            'Cambio medio en Mediana': f'{np.mean(cambios_mediana):.1f}%',
            'Max cambio en Media': f'{np.max(cambios_media):.1f}%'
        })
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sintetico_cisne_negro.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Guardado: {OUTPUT_DIR / 'sintetico_cisne_negro.png'}")
    
    # Mostrar tabla
    print("\n   Impacto de UN cisne negro:")
    for r in resultados:
        print(f"   {r['Distribución']}:")
        print(f"      Cambio en media: {r['Cambio medio en Media']} (max: {r['Max cambio en Media']})")
        print(f"      Cambio en mediana: {r['Cambio medio en Mediana']}")
    
    return resultados


# =============================================================================
# PARTE C: CRITERIO KAPPA DE TALEB
# =============================================================================

def experimento_kappa(sample_sizes=[100, 500, 1000, 5000], n_simulations=500):
    """
    Calcula κ = max(X) / sum(X) para diferentes distribuciones y tamaños.
    """
    print("\n" + "="*60)
    print("PARTE C: Criterio κ de Taleb")
    print("="*60)
    
    distribuciones = [
        ('Normal |X|', lambda n: np.abs(np.random.normal(0, 1, n)), COLORS['normal']),
        ('Pareto(α=3)', lambda n: stats.pareto.rvs(3, size=n), COLORS['pareto3']),
        ('Pareto(α=1.5)', lambda n: stats.pareto.rvs(1.5, size=n), COLORS['pareto15']),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (nombre, sampler, color) in zip(axes, distribuciones):
        for n in sample_sizes:
            kappas = []
            for _ in range(n_simulations):
                data = sampler(n)
                kappa = data.max() / data.sum()
                kappas.append(kappa)
            
            ax.hist(kappas, bins=30, alpha=0.5, label=f'n={n}', density=True)
        
        ax.set_xlabel('κ = max(X) / sum(X)')
        ax.set_ylabel('Densidad')
        ax.set_title(nombre)
        ax.legend()
        
        # Línea de referencia: 1/n para la muestra más grande
        ax.axvline(x=1/sample_sizes[-1], color='gray', linestyle=':', 
                  label=f'1/n = {1/sample_sizes[-1]:.4f}')
    
    plt.suptitle('Criterio κ de Taleb: ¿Una observación domina el total?', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sintetico_kappa.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Guardado: {OUTPUT_DIR / 'sintetico_kappa.png'}")
    
    # Calcular κ promedio para n=1000
    print("\n   κ promedio para n=1000:")
    for nombre, sampler, _ in distribuciones:
        kappas = [sampler(1000).max() / sampler(1000).sum() for _ in range(500)]
        print(f"   {nombre}: κ = {np.mean(kappas):.4f}")


# =============================================================================
# PARTE D: CURVA DE CONCENTRACIÓN (LORENZ)
# =============================================================================

def experimento_concentracion(n=10000):
    """
    Calcula qué porcentaje del total aporta el top X%.
    """
    print("\n" + "="*60)
    print("PARTE D: Curva de Concentración (Lorenz)")
    print("="*60)
    
    distribuciones = [
        ('Normal |X|', np.abs(np.random.normal(0, 1, n)), COLORS['normal']),
        ('Pareto(α=2.5)', stats.pareto.rvs(2.5, size=n), COLORS['pareto3']),
        ('Pareto(α=1.5)', stats.pareto.rvs(1.5, size=n), COLORS['pareto15']),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    resultados = []
    
    for nombre, data, color in distribuciones:
        # Ordenar de mayor a menor
        sorted_data = np.sort(data)[::-1]
        cumsum = np.cumsum(sorted_data) / sorted_data.sum()
        percentiles = np.arange(1, n+1) / n * 100
        
        ax.plot(percentiles, cumsum * 100, linewidth=2, label=nombre, color=color)
        
        # Calcular concentración
        top_1 = cumsum[int(n * 0.01)] * 100
        top_10 = cumsum[int(n * 0.10)] * 100
        top_20 = cumsum[int(n * 0.20)] * 100
        
        resultados.append({
            'Distribución': nombre,
            'Top 1%': f'{top_1:.1f}%',
            'Top 10%': f'{top_10:.1f}%',
            'Top 20%': f'{top_20:.1f}%'
        })
    
    # Línea de igualdad perfecta
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Igualdad perfecta')
    
    ax.set_xlabel('% de observaciones (de mayor a menor)')
    ax.set_ylabel('% del total acumulado')
    ax.set_title('Curva de Concentración: ¿Cuánto aporta el top X%?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sintetico_concentracion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Guardado: {OUTPUT_DIR / 'sintetico_concentracion.png'}")
    
    # Mostrar tabla
    print("\n   Concentración del total:")
    print(f"   {'Distribución':<20} {'Top 1%':>10} {'Top 10%':>10} {'Top 20%':>10}")
    print("   " + "-"*52)
    for r in resultados:
        print(f"   {r['Distribución']:<20} {r['Top 1%']:>10} {r['Top 10%']:>10} {r['Top 20%']:>10}")
    
    return resultados


# =============================================================================
# BONUS: CAUCHY - CUANDO LA MEDIA NO EXISTE
# =============================================================================

def experimento_cauchy(n_max=10000, n_runs=20):
    """
    Demuestra que el promedio de Cauchy no converge.
    """
    print("\n" + "="*60)
    print("BONUS: Cauchy - Cuando la Media No Existe")
    print("="*60)
    
    ns = np.arange(1, n_max + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Normal
    ax = axes[0]
    for _ in range(n_runs):
        data = np.random.normal(0, 1, n_max)
        running_mean = np.cumsum(data) / ns
        ax.plot(ns, running_mean, alpha=0.4, linewidth=0.5, color=COLORS['normal'])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('n')
    ax.set_ylabel('Promedio acumulado')
    ax.set_title('Normal(0,1): CONVERGE a μ=0')
    ax.set_ylim(-1, 1)
    
    # Cauchy
    ax = axes[1]
    for _ in range(n_runs):
        data = stats.cauchy.rvs(size=n_max)
        running_mean = np.cumsum(data) / ns
        ax.plot(ns, running_mean, alpha=0.4, linewidth=0.5, color=COLORS['cauchy'])
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('n')
    ax.set_ylabel('Promedio acumulado')
    ax.set_title('Cauchy(0,1): NO CONVERGE\n(La media no existe)')
    
    # Anotación
    ax.annotate('El promedio de n Cauchy\nes también Cauchy!', 
               xy=(0.5, 0.05), xycoords='axes fraction',
               ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sintetico_cauchy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Guardado: {OUTPUT_DIR / 'sintetico_cauchy.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Ejecuta todos los experimentos."""
    print("="*60)
    print("EJERCICIO: Anatomía de las Colas Largas")
    print("="*60)
    
    # Parte A
    experimento_convergencia()
    
    # Parte B
    experimento_cisne_negro()
    
    # Parte C
    experimento_kappa()
    
    # Parte D
    experimento_concentracion()
    
    # Bonus
    experimento_cauchy()
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print("""
    Lecciones clave:
    
    1. CONVERGENCIA: En fat tails (α ≤ 2), el promedio converge 
       tan lentamente que es prácticamente inútil.
    
    2. FRAGILIDAD: En fat tails, UNA observación puede cambiar
       drásticamente la media. La mediana es robusta.
    
    3. CONCENTRACIÓN: En fat tails, el top 10% puede tener más
       del 90% del total ("winner takes all").
    
    4. CAUCHY: Caso extremo donde la media NO EXISTE.
       El promedio de n Cauchy es Cauchy.
    
    ¿Estás en Mediocristán o Extremistán?
    """)
    
    print(f"\n📁 Resultados guardados en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
