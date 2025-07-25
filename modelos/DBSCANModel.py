import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from kneed import KneeLocator


class DBSCANModel:
    """
    Pipeline para clustering no supervisado con DBSCAN.
    - Permite aplicar PCA para reducción de dimensionalidad.
    - Incluye gráfico k-dist para estimar epsilon.
    - Evalúa la calidad de los clusters con silhouette score.
    - Visualiza los clusters en 2D.
    - Selecciona automáticamente columnas numéricas si no se especifican.
    - Permite ajustar parámetros como eps y min_samples.
    """

    def __init__(self, df, features=None, k=9, eps=None, min_samples=50):
        """
        Args:
            df (pd.DataFrame): DataFrame con las variables de entrada.
            features (list): Lista de columnas a usar para clustering (si es None, selecciona numéricas automáticamente)
            k (int): Número de vecinos para el gráfico k-dist.
            eps (float): Valor epsilon para DBSCAN. Si None, se estima automáticamente.
            min_samples (int): Número mínimo de muestras para formar un clúster.
        """
        self.df = df.copy()
        self.all_columns = df.columns.tolist()

        # Selección automática de columnas numéricas excluyendo columnas irrelevantes
        default_features = df.select_dtypes(include=[np.number]).columns.tolist()
        default_features = [col for col in default_features
                            if col not in ['paciente_id', 'timestamp', 'fase_ela', 'estado', 'empeoramiento']]

        self.features = features if features else default_features

        # Guardar subset de features numéricas
        self.df = self.df[self.features]

        # Parámetros de clustering
        self.k = k
        self.eps = eps
        self.min_samples = min_samples

        # Matriz de datos lista para clustering
        self.X = self.df.dropna().values

        # Modelos y resultados
        self.model = None
        self.labels_ = None
        self.clustered_df = None

    def prepare_data_for_clustering(self, n_components=10):
        """
        Prepara los datos para clustering:
        - Elimina columnas con '_etiqueta'.
        - Aplica PCA para reducir a `n_components`.

        Args:
            n_components (int): Número de componentes principales para PCA.
        """
        # Eliminar columnas irrelevantes con "_etiqueta"
        cols_to_drop = [col for col in self.df.columns if col.endswith("_etiqueta")]
        if cols_to_drop:
            print(f"Columnas eliminadas: {cols_to_drop}")
            self.df = self.df.drop(columns=cols_to_drop)

        # Seleccionar columnas numéricas
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.features = [col for col in numeric_features if
                         col not in ['paciente_id', 'timestamp', 'fase_ela', 'estado', 'empeoramiento']]

        # Aplicar PCA
        print(f"Aplicando PCA para reducir a {n_components} componentes...")
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.df[self.features])

        # Guardar los datos transformados
        self.df = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])])
        self.X = self.df.values

        print(f"Shape tras PCA: {self.df.shape}")
        return self.df

    def plot_k_distance(self, auto_eps=False, show_knee=True):
        """
        Genera la gráfica de distancias al k-ésimo vecino para sugerir epsilon.
        Args:
            auto_eps (bool): Si True, calcula automáticamente epsilon usando el knee del gráfico.
            show_knee (bool): Si True, muestra el knee en la gráfica.
        """
        neighbors = NearestNeighbors(n_neighbors=self.k)
        neighbors_fit = neighbors.fit(self.X)
        distances, _ = neighbors_fit.kneighbors(self.X)
        distances = np.sort(distances[:, self.k - 1])

        knee = None
        if show_knee:
            knee_locator = KneeLocator(range(len(distances)), distances,
                                       curve="convex", direction="increasing")
            knee = knee_locator.knee
            if knee is not None:
                print(f"Epsilon sugerido automáticamente (knee): {distances[knee]:.4f}")

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=np.arange(len(distances)), y=distances, linewidth=2, color="blue")
        plt.xlabel("Puntos ordenados")
        plt.ylabel(f"Distancia al {self.k}-ésimo vecino")
        plt.title("Gráfico k-dist para encontrar epsilon")
        plt.grid(True, linestyle="--", alpha=0.7)

        if knee is not None:
            plt.axvline(knee, color="red", linestyle="--",
                        label=f"Knee (ε ≈ {distances[knee]:.3f})")
            plt.legend()

        plt.show()

        if auto_eps and knee is not None:
            self.eps = float(distances[knee])
            print(f"Epsilon asignado automáticamente: {self.eps:.4f}")

        return distances

    def fit(self, epsilon=None, auto_eps=True):
        """
        Ajusta el modelo DBSCAN a los datos preparados.

        Args:
            epsilon (float): Valor de epsilon para DBSCAN. Si None, se estima automáticamente.
            auto_eps (bool): Si True y no hay epsilon, se calcula automáticamente.
        """
        if epsilon is not None:
            self.eps = epsilon
            print(f"Usando epsilon proporcionado: {self.eps:.4f}")

        if self.eps is None and auto_eps:
            print("[No se ha definido epsilon. Calculando automáticamente...")
            self.plot_k_distance(auto_eps=True)
        elif self.eps is None:
            print("ERROR: No se ha definido epsilon. Ejecuta `plot_k_distance()` o pasa un valor.")
            return None

        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.model.fit_predict(self.X)

        self.clustered_df = self.df.copy()
        self.clustered_df["cluster"] = self.labels_
        return self.clustered_df

    def evaluate(self):
        """
        Calcula el silhouette score para evaluar la calidad de los clusters.
        Excluye puntos de ruido (-1) del cálculo.
        """
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
        """
        Visualiza los clusters en un plano 2D usando dos variables.
        Args:
            x_feature (str): Nombre de la columna para el eje X.
            y_feature (str): Nombre de la columna para el eje Y.
        """
        if self.clustered_df is None:
            print("[ERROR] Ajusta el modelo con `fit()` antes de visualizar.")
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
