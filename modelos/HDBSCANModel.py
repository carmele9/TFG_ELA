import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import hdbscan


class HDBSCANModel:
    """
    Pipeline para clustering no supervisado con HDBSCAN.
    - Permite evaluar la calidad de los clusters con silhouette score.
    - Incluye preprocesamiento de datos eliminando columnas '_etiqueta' y aplicando PCA.
    - Visualiza los clusters en 2D y muestra puntuaciones de outlier.
    - Permite ajustar parámetros como min_cluster_size, min_samples y metric.
    - Selecciona automáticamente columnas numéricas si no se especifican.
    """

    def __init__(self, df, features=None, min_cluster_size=50, min_samples=None, metric='euclidean'):
        """
        Args:
            df (pd.DataFrame): DataFrame con las variables de entrada.
            features (list): Lista de columnas a usar para clustering.
            min_cluster_size (int): Tamaño mínimo de un cluster.
            min_samples (int): Número mínimo de muestras para que un punto sea núcleo.
            metric (str): Métrica de distancia para clustering.
        """
        self.df = df.copy()
        self.all_columns = df.columns.tolist()

        # Selección automática de columnas numéricas
        default_features = df.select_dtypes(include=[np.number]).columns.tolist()
        default_features = [col for col in default_features
                            if col not in ['paciente_id', 'timestamp', 'fase_ela', 'estado', 'empeoramiento']]

        self.features = features if features else default_features

        # Filtrar solo las features relevantes
        self.df = self.df[self.features]
        self.X = self.df.dropna().values

        # Parámetros del modelo
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

        # Modelos y resultados
        self.model = None
        self.labels_ = None
        self.clustered_df = None

    def prepare_data_for_clustering(self, n_components=10):
        """
        Prepara los datos para clustering:
        - Elimina columnas cuyo nombre termine con '_etiqueta'.
        - Selecciona solo columnas numéricas.
        - Aplica PCA para reducir la dimensionalidad.

        Args:
            n_components (int): Número de componentes principales para PCA.
        """
        # Eliminar columnas con sufijo '_etiqueta'
        cols_to_drop = [col for col in self.df.columns if col.endswith("_etiqueta")]
        if cols_to_drop:
            print(f"Columnas eliminadas: {cols_to_drop}")
            self.df = self.df.drop(columns=cols_to_drop)

        # Selección de columnas numéricas
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

    def fit(self, min_cluster_size=None, min_samples=None, metric=None):
        """
        Ajusta el modelo HDBSCAN.

        Args:
            min_cluster_size (int): Tamaño mínimo de un cluster (opcional).
            min_samples (int): Número mínimo de muestras núcleo (opcional).
            metric (str): Métrica de distancia (opcional).
        """
        if min_cluster_size is not None:
            self.min_cluster_size = min_cluster_size
        if min_samples is not None:
            self.min_samples = min_samples
        if metric is not None:
            self.metric = metric

        # Ajustar HDBSCAN
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric
        )
        self.labels_ = self.model.fit_predict(self.X)

        self.clustered_df = self.df.copy()
        self.clustered_df["cluster"] = self.labels_
        print(f"Clustering completado. Se detectaron "
              f"{len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)} clusters.")
        return self.clustered_df

    def evaluate(self):
        """
        Calcula el silhouette score para evaluar la calidad de los clusters (excluyendo ruido).
        """
        if self.labels_ is None:
            print("ERROR: Ajusta el modelo con `fit()` antes de evaluar.")
            return None

        unique_labels = set(self.labels_) - {-1}
        if len(unique_labels) <= 1:
            print("Solo se detectó un cluster o ruido. Silhouette no aplicable.")
            return None

        mask = self.labels_ != -1
        score = silhouette_score(self.X[mask], self.labels_[mask])
        print(f"Silhouette Score (sin ruido): {score:.4f}")
        return score

    def plot_clusters(self, x_feature, y_feature):
        """
        Visualiza los clusters en un plano 2D usando dos variables.
        Args:
            x_feature (str): Nombre de la columna para el eje X.
            y_feature (str): Nombre de la columna para el eje Y.
        """
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
        plt.title(f"Clusters HDBSCAN: {x_feature} vs {y_feature}")
        plt.legend(title="Cluster")
        plt.show()

    def plot_outlier_scores(self):
        """
        Muestra un histograma de las puntuaciones de outlier asignadas por HDBSCAN.
        """
        if self.model is None or not hasattr(self.model, 'outlier_scores_'):
            print("ERROR: Ajusta el modelo con `fit()` antes de visualizar los outliers.")
            return

        plt.figure(figsize=(8, 5))
        sns.histplot(self.model.outlier_scores_, bins=50, kde=True)
        plt.title("Distribución de puntuaciones de outlier (HDBSCAN)")
        plt.xlabel("Outlier score")
        plt.ylabel("Frecuencia")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
