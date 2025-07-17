import time
from memory_profiler import memory_usage
import matplotlib.pyplot as plt


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

        libs = list(self.results.keys())
        tiempos = [self.results[lib]['time_sec'] for lib in libs]
        memoria = [self.results[lib]['max_memory_mb'] for lib in libs]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_time = 'tab:blue'
        ax1.set_xlabel('Librería')
        ax1.set_ylabel('Tiempo (segundos)', color=color_time)
        bars_time = ax1.bar(libs, tiempos, color=color_time, alpha=0.6, label='Tiempo (s)')
        ax1.tick_params(axis='y', labelcolor=color_time)
        ax1.set_ylim(0, max(tiempos) * 1.3)

        ax2 = ax1.twinx()
        color_mem = 'tab:green'
        ax2.set_ylabel('Memoria (MB)', color=color_mem)
        bars_mem = ax2.bar(libs, memoria, color=color_mem, alpha=0.6, label='Memoria (MB)', width=0.4)
        ax2.tick_params(axis='y', labelcolor=color_mem)
        ax2.set_ylim(0, max(memoria) * 1.3)
        ax2.bar_label(bars_mem, padding=3, fmt='%.2f')

        # Añadir etiquetas de tiempo encima de las barras
        ax1.bar_label(bars_time, padding=3, fmt='%.2f')

        plt.title('Comparación de rendimiento entre Pandas y Polars')
        fig.tight_layout()
        plt.show()
