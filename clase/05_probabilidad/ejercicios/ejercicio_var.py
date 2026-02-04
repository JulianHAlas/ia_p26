#!/usr/bin/env python3
"""
Ejercicio: El Fraude del Value at Risk (VaR)

Este script demuestra por qué el VaR falla cuando los retornos
tienen colas pesadas.

Ejecutar:
    python ejercicio_var.py

Autor: Módulo de Probabilidad - IA P26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Crear directorio de outputs
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def descargar_datos(ticker="^GSPC", start="1990-01-01"):
    """Descarga datos históricos."""
    print(f"📥 Descargando datos de {ticker}...")
    
    data = yf.download(ticker, start=start, progress=False)
    data['Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data = data.dropna()
    
    print(f"   ✓ {len(data)} observaciones")
    return data


def calcular_var_normal(returns, alpha=0.99):
    """
    Calcula VaR paramétrico asumiendo normalidad.
    
    VaR_alpha = -(mu + z_{1-alpha} * sigma)
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - alpha)  # Cuantil negativo
    var = -(mu + z * sigma)
    return var


def calcular_var_t(returns, alpha=0.99, df=5):
    """
    Calcula VaR usando distribución Student-t.
    """
    mu = returns.mean()
    sigma = returns.std()
    # Ajustar escala para Student-t
    scale = sigma * np.sqrt((df - 2) / df) if df > 2 else sigma
    t_quantile = stats.t.ppf(1 - alpha, df)
    var = -(mu + t_quantile * scale)
    return var


def calcular_var_historico(returns, alpha=0.99):
    """
    Calcula VaR usando simulación histórica (no paramétrico).
    """
    return -np.percentile(returns, (1 - alpha) * 100)


def backtest_var(data, window=252, alpha=0.99, method='normal', df=5):
    """
    Realiza backtesting del VaR.
    
    Args:
        data: DataFrame con columna 'Returns'
        window: Ventana móvil para estimar parámetros
        alpha: Nivel de confianza (0.99 = VaR 99%)
        method: 'normal', 't', o 'historical'
        df: Grados de libertad si method='t'
    
    Returns:
        DataFrame con VaR predicho, retorno real, y si hubo violación
    """
    returns = data['Returns']
    results = []
    
    for i in range(window, len(returns)):
        # Datos históricos hasta el día anterior
        hist_returns = returns.iloc[i-window:i]
        
        # Calcular VaR
        if method == 'normal':
            var = calcular_var_normal(hist_returns, alpha)
        elif method == 't':
            var = calcular_var_t(hist_returns, alpha, df)
        else:  # historical
            var = calcular_var_historico(hist_returns, alpha)
        
        # Retorno real del día siguiente
        actual_return = returns.iloc[i]
        
        # ¿Hubo violación?
        violation = actual_return < -var
        
        results.append({
            'Date': returns.index[i],
            'VaR': var,
            'Actual_Return': actual_return,
            'Violation': violation,
            'Exceedance': (-actual_return - var) if violation else 0
        })
    
    return pd.DataFrame(results)


def analizar_violaciones(backtest_results, alpha=0.99):
    """
    Analiza las violaciones del VaR.
    """
    n_total = len(backtest_results)
    n_violations = backtest_results['Violation'].sum()
    
    expected_rate = 1 - alpha
    actual_rate = n_violations / n_total
    
    # Severidad de violaciones
    violations = backtest_results[backtest_results['Violation']]
    
    if len(violations) > 0:
        avg_exceedance = violations['Exceedance'].mean()
        max_exceedance = violations['Exceedance'].max()
        avg_var = violations['VaR'].mean()
        
        # Expected Shortfall empírico
        es = -violations['Actual_Return'].mean()
        es_ratio = es / avg_var if avg_var > 0 else np.nan
    else:
        avg_exceedance = max_exceedance = es = es_ratio = 0
    
    return {
        'Total Days': n_total,
        'Violations': n_violations,
        'Expected Rate': f'{expected_rate:.2%}',
        'Actual Rate': f'{actual_rate:.2%}',
        'Ratio (Actual/Expected)': f'{actual_rate/expected_rate:.2f}x',
        'Avg Exceedance': f'{avg_exceedance*100:.2f}%',
        'Max Exceedance': f'{max_exceedance*100:.2f}%',
        'Expected Shortfall': f'{es*100:.2f}%',
        'ES/VaR Ratio': f'{es_ratio:.2f}x' if not np.isnan(es_ratio) else 'N/A'
    }


def encontrar_mejor_df(data, window=252, alpha=0.99, df_range=range(2, 31)):
    """
    Encuentra los grados de libertad óptimos para Student-t.
    """
    results = []
    expected_rate = 1 - alpha
    
    for df in df_range:
        backtest = backtest_var(data, window, alpha, method='t', df=df)
        actual_rate = backtest['Violation'].sum() / len(backtest)
        error = abs(actual_rate - expected_rate)
        results.append({'df': df, 'actual_rate': actual_rate, 'error': error})
    
    results_df = pd.DataFrame(results)
    best_df = results_df.loc[results_df['error'].idxmin(), 'df']
    
    return int(best_df), results_df


def plot_var_backtest(backtest_normal, backtest_t, alpha=0.99):
    """
    Visualiza el backtest del VaR.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Retornos y VaR Normal
    ax = axes[0, 0]
    ax.plot(backtest_normal['Date'], backtest_normal['Actual_Return'] * 100, 
            alpha=0.7, linewidth=0.5, label='Retornos')
    ax.plot(backtest_normal['Date'], -backtest_normal['VaR'] * 100, 
            'r-', linewidth=1, label=f'-VaR {alpha:.0%}')
    
    # Marcar violaciones
    violations = backtest_normal[backtest_normal['Violation']]
    ax.scatter(violations['Date'], violations['Actual_Return'] * 100, 
               color='red', s=20, zorder=5, label='Violaciones')
    
    ax.set_ylabel('Retorno (%)')
    ax.set_title(f'VaR Normal: {len(violations)} violaciones '
                f'({len(violations)/len(backtest_normal):.1%})')
    ax.legend(loc='lower left')
    
    # Panel 2: Retornos y VaR Student-t
    ax = axes[0, 1]
    ax.plot(backtest_t['Date'], backtest_t['Actual_Return'] * 100, 
            alpha=0.7, linewidth=0.5, label='Retornos')
    ax.plot(backtest_t['Date'], -backtest_t['VaR'] * 100, 
            'g-', linewidth=1, label=f'-VaR {alpha:.0%}')
    
    violations_t = backtest_t[backtest_t['Violation']]
    ax.scatter(violations_t['Date'], violations_t['Actual_Return'] * 100, 
               color='red', s=20, zorder=5, label='Violaciones')
    
    ax.set_ylabel('Retorno (%)')
    ax.set_title(f'VaR Student-t: {len(violations_t)} violaciones '
                f'({len(violations_t)/len(backtest_t):.1%})')
    ax.legend(loc='lower left')
    
    # Panel 3: Distribución de exceedances (Normal)
    ax = axes[1, 0]
    if len(violations) > 0:
        exceedances = violations['Exceedance'] * 100
        ax.hist(exceedances, bins=30, alpha=0.7, color='coral', edgecolor='white')
        ax.axvline(x=exceedances.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Media: {exceedances.mean():.2f}%')
        ax.set_xlabel('Exceso sobre VaR (%)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Severidad de Violaciones (Normal)\nCuando falla, ¿por cuánto?')
        ax.legend()
    
    # Panel 4: Comparación de tasas de violación
    ax = axes[1, 1]
    expected = (1 - alpha) * 100
    actual_normal = len(violations) / len(backtest_normal) * 100
    actual_t = len(violations_t) / len(backtest_t) * 100
    
    bars = ax.bar(['Esperado', 'Normal', 'Student-t'], 
                  [expected, actual_normal, actual_t],
                  color=['green', 'coral', 'steelblue'])
    ax.axhline(y=expected, color='green', linestyle='--', linewidth=2)
    ax.set_ylabel('Tasa de Violación (%)')
    ax.set_title(f'Comparación: Tasa de Violación VaR {alpha:.0%}')
    
    # Añadir valores sobre las barras
    for bar, val in zip(bars, [expected, actual_normal, actual_t]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{val:.2f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'var_backtest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Guardado: {OUTPUT_DIR / 'var_backtest.png'}")


def plot_df_optimization(df_results, alpha=0.99):
    """
    Visualiza la búsqueda de grados de libertad óptimos.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    expected = 1 - alpha
    
    ax.plot(df_results['df'], df_results['actual_rate'] * 100, 'bo-', linewidth=2)
    ax.axhline(y=expected * 100, color='green', linestyle='--', linewidth=2, 
              label=f'Tasa esperada ({expected:.0%})')
    
    best_df = df_results.loc[df_results['error'].idxmin(), 'df']
    best_rate = df_results.loc[df_results['error'].idxmin(), 'actual_rate']
    ax.scatter([best_df], [best_rate * 100], color='red', s=100, zorder=5,
              label=f'Óptimo: df={best_df}')
    
    ax.set_xlabel('Grados de libertad (ν)')
    ax.set_ylabel('Tasa de violación (%)')
    ax.set_title('Búsqueda del mejor ajuste Student-t\n'
                '(ν pequeño = colas más pesadas)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'var_df_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Guardado: {OUTPUT_DIR / 'var_df_optimization.png'}")


def main():
    """Ejecuta el análisis completo de VaR."""
    print("="*60)
    print("EJERCICIO: El Fraude del Value at Risk (VaR)")
    print("="*60)
    
    # 1. Descargar datos
    data = descargar_datos()
    
    # 2. Parámetros
    window = 252  # 1 año de datos para estimar
    alpha = 0.99  # VaR 99%
    
    print(f"\n📊 Parámetros:")
    print(f"   Ventana de estimación: {window} días")
    print(f"   Nivel de confianza: {alpha:.0%}")
    
    # 3. Backtest VaR Normal
    print(f"\n⏳ Calculando VaR Normal...")
    backtest_normal = backtest_var(data, window, alpha, method='normal')
    
    # 4. Encontrar mejores grados de libertad para Student-t
    print(f"⏳ Buscando mejor ajuste Student-t...")
    best_df, df_results = encontrar_mejor_df(data, window, alpha)
    print(f"   ✓ Mejor df: {best_df}")
    
    # 5. Backtest VaR Student-t
    print(f"⏳ Calculando VaR Student-t (df={best_df})...")
    backtest_t = backtest_var(data, window, alpha, method='t', df=best_df)
    
    # 6. Analizar resultados
    print("\n" + "="*60)
    print(f"RESULTADOS: VaR {alpha:.0%}")
    print("="*60)
    
    print("\n📉 VaR Normal (asume normalidad):")
    analysis_normal = analizar_violaciones(backtest_normal, alpha)
    for key, value in analysis_normal.items():
        print(f"   {key}: {value}")
    
    print(f"\n📈 VaR Student-t (df={best_df}):")
    analysis_t = analizar_violaciones(backtest_t, alpha)
    for key, value in analysis_t.items():
        print(f"   {key}: {value}")
    
    # 7. Generar visualizaciones
    print("\n" + "="*60)
    print("GENERANDO VISUALIZACIONES")
    print("="*60)
    
    plot_var_backtest(backtest_normal, backtest_t, alpha)
    plot_df_optimization(df_results, alpha)
    
    # 8. Conclusión
    print("\n" + "="*60)
    print("CONCLUSIÓN")
    print("="*60)
    
    normal_rate = backtest_normal['Violation'].sum() / len(backtest_normal)
    t_rate = backtest_t['Violation'].sum() / len(backtest_t)
    expected_rate = 1 - alpha
    
    print(f"""
    VaR {alpha:.0%} - Comparación de modelos:
    
    Tasa de violación esperada: {expected_rate:.2%}
    
    • VaR Normal:    {normal_rate:.2%} ({normal_rate/expected_rate:.1f}x más de lo esperado)
    • VaR Student-t: {t_rate:.2%} ({t_rate/expected_rate:.1f}x más de lo esperado)
    
    El modelo normal subestima las violaciones por un factor de {normal_rate/expected_rate:.1f}x.
    
    Peor aún: cuando el VaR falla, la pérdida real es mucho mayor que el VaR predicho.
    Esto es exactamente lo que pasó en 2008.
    
    Lección: No confíes ciegamente en modelos de riesgo basados en normalidad.
    """)
    
    print(f"\n📁 Resultados guardados en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
