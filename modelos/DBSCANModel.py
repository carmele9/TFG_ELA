import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


class DBSCANModel:
    """
    Pipeline para clustering no supervisado con DBSCAN.
    - Incluye gráfico k-dist para estimar epsilon.
    - Evalúa la calidad de los clusters con silhouette score.
    """

    def __init__(self, df, features=None, k=9, eps=None, min_samples=50):
        """
        Args:
            df (pd.DataFrame): DataFrame con las variables de entrada.
            features (list): Lista de columnas a usar para clustering.
            k (int): Número de vecinos para el gráfico k-dist.
            eps (float): Valor epsilon para DBSCAN. Si None, se puede estimar con k-dist.
            min_samples (int): Número mínimo de muestras para formar un clúster.
        """
        self.df = df.copy()
        self.all_columns = df.columns.tolist()
        default_features = df.select_dtypes(include=[np.number]).columns.tolist()
        default_features = [col for col in default_features if
                            col not in ['paciente_id', 'timestamp', 'fase_ela', 'estado', 'empeoramiento']]
        self.features = features if features else default_features

        # Filtrar solo las features en el DataFrame interno
        self.df = self.df[self.features]
        self.k = k
        self.eps = eps
        self.min_samples = min_samples
        self.X = self.df[self.features].dropna().values

        self.model = None
        self.labels_ = None
        self.clustered_df = None

    def plot_k_distance(self, auto_eps=False, show_knee=True):
        """
        Genera la gráfica de distancias al k-ésimo vecino para sugerir epsilon.
        - Dibuja grid.
        - Puede detectar el "codo" automáticamente con Kneedle.
        """
        neighbors = NearestNeighbors(n_neighbors=self.k)
        neighbors_fit = neighbors.fit(self.X)
        distances, _ = neighbors_fit.kneighbors(self.X)
        distances = np.sort(distances[:, self.k - 1])

        # Detección automática del "codo" (knee point)
        knee = None
        if show_knee:
            knee_locator = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
            knee = knee_locator.knee
            if knee is not None:
                print(f"INFO: Epsilon sugerido automáticamente (knee): {distances[knee]:.4f}")

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=np.arange(len(distances)), y=distances, linewidth=2, color="blue")
        plt.xlabel("Puntos ordenados")
        plt.ylabel(f"Distancia al {self.k}-ésimo vecino")
        plt.title("Gráfico k-dist para encontrar epsilon")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Marcar el knee si existe
        if knee is not None:
            plt.axvline(knee, color="red", linestyle="--", label=f"Knee (ε ≈ {distances[knee]:.3f})")
            plt.legend()

        plt.show()

        if auto_eps and knee is not None:
            self.eps = float(distances[knee])
            print(f"INFO: Epsilon asignado automáticamente: {self.eps:.4f}")

        return distances

    def fit(self, epsilon=None, auto_eps=True):
        """
        Ajusta el modelo DBSCAN.
        Args:
            epsilon (float): Valor opcional de epsilon. Si se pasa, sobrescribe el actual.
            auto_eps (bool): Si True y no hay epsilon, lo define automáticamente.
        """
        if epsilon is not None:
            self.eps = epsilon
            print(f"Usando epsilon proporcionado: {self.eps:.4f}")

        if self.eps is None and auto_eps:
            print("No se ha definido epsilon. Calculando automáticamente...")
            distances = self.plot_k_distance(auto_eps=True)
        elif self.eps is None:
            print("ERROR: No se ha definido epsilon. Ejecuta `plot_k_distance()` o pasa un valor.")
            return None

        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.model.fit_predict(self.X)

        self.clustered_df = self.df.copy()
        self.clustered_df["cluster"] = self.labels_
        return self.clustered_df

    def evaluate(self):
        """Calcula el silhouette score para evaluar la calidad de los clusters."""
        if self.labels_ is None:
            print("ERROR: Ajusta el modelo con `fit()` antes de evaluar.")
            return None

        if len(set(self.labels_)) <= 1:
            print("Solo se detectó un cluster o ruido. Silhouette no aplicable.")
            return None

        score = silhouette_score(self.X, self.labels_)
        print(f"Silhouette Score: {score:.4f}")
        return score

    def plot_clusters(self, x_feature, y_feature):
        """Muestra los clusters en un plano 2D usando dos variables."""
        if self.clustered_df is None:
            print("ERROR: Ajusta el modelo con `fit()` antes de visualizar.")
            return

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=self.clustered_df,
            x=x_feature,
            y=y_feature,
            hue="cluster",
            palette="tab10",
            alpha=0.7
        )
        plt.title(f"Clusters DBSCAN: {x_feature} vs {y_feature}")
        plt.legend(title="Cluster")
        plt.show()