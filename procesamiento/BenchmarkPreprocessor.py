import time
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import numpy as np


class BenchmarkPreprocessor:
    """
    Clase para comparar tiempos y memoria usados por dos preprocesadores
    (pandas y polars) sobre datasets dados.
    """

    def __init__(self, pandas_class, polars_class, df_pandas, df_polars, generar_secuencias=True):
        """
        Inicializa la clase con los preprocesadores y datasets.

        Args:
            pandas_class: Clase del preprocesador basado en pandas.
            polars_class: Clase del preprocesador basado en polars.
            df_pandas: DataFrame pandas.
            df_polars: DataFrame polars.
            generar_secuencias (bool): Si generar secuencias tipo LSTM.
        """
        self.pandas_class = pandas_class
        self.polars_class = polars_class
        self.df_pandas = df_pandas
        self.df_polars = df_polars
        self.generar_secuencias = generar_secuencias
        self.results = {}

    def _run_pandas(self):
        pre = self.pandas_class(self.df_pandas)
        if self.generar_secuencias:
            return pre.run_all(generar_secuencias=True)
        else:
            return pre.run_all()

    def _run_polars(self):
        pre = self.polars_class(self.df_polars)
        if self.generar_secuencias:
            return pre.run_all(generar_secuencias=True)
        else:
            return pre.run_all()

    def run_benchmark(self):
        """Ejecuta el benchmark y guarda los resultados de tiempo y memoria."""

        print("Ejecutando PandasPreprocessor...")
        start = time.time()
        mem_pandas = memory_usage((self._run_pandas,), interval=0.1)
        time_pandas = time.time() - start
        self.results['pandas'] = {
            'time_sec': round(time_pandas, 4),
            'max_memory_mb': round(max(mem_pandas) - min(mem_pandas), 4)
        }

        print("Ejecutando PolarsPreprocessor...")
        start = time.time()
        mem_polars = memory_usage((self._run_polars,), interval=0.1)
        time_polars = time.time() - start
        self.results['polars'] = {
            'time_sec': round(time_polars, 4),
            'max_memory_mb': round(max(mem_polars) - min(mem_polars), 4)
        }

        print("\nBenchmarking completado:")
        for lib, res in self.results.items():
            print(f"➡ {lib.capitalize()}: {res['time_sec']}s, {res['max_memory_mb']}MB")

    def get_results(self):
        """Devuelve los resultados del benchmarking."""
        return self.results

    def plot_results(self):
        """Genera un gráfico comparativo de tiempos y memoria usados por cada preprocesador."""
        if not self.results:
            print("No hay resultados para mostrar. Ejecuta run_benchmark() primero.")
            return
        # Obtener los resultados
        libs = list(self.results.keys())
        tiempos = [self.results[lib]['time_sec'] for lib in libs]
        memoria = [self.results[lib]['max_memory_mb'] for lib in libs]

        x = np.arange(len(libs))
        width = 0.35

        # Colores personalizados para las barras
        color_tiempo = "#FF7F50"
        color_memoria = "#6495ED"

        fig, ax = plt.subplots(figsize=(9, 5))
        # Gráfico de barras para tiempos y memoria
        bars_time = ax.bar(
            x - width / 2, tiempos, width,
            label='Tiempo (s)', color=color_tiempo,
            alpha=0.85, edgecolor="black", linewidth=1
        )
        bars_mem = ax.bar(
            x + width / 2, memoria, width,
            label='Memoria (MB)', color=color_memoria,
            alpha=0.85, edgecolor="black", linewidth=1
        )

        # Personalización del gráfico
        ax.set_facecolor("#f9f9f9")
        fig.patch.set_facecolor("#f2f2f2")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Etiquetas y título
        ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de rendimiento entre Pandas y Polars',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([lib.capitalize() for lib in libs], fontsize=11, fontweight='medium')
        ax.legend(facecolor='white', framealpha=1, edgecolor='lightgray')

        # Etiquetas sobre las barras
        for bar in bars_time:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        # Anotaciones para las barras de memoria
        for bar in bars_mem:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

        fig.tight_layout()
        plt.show()
