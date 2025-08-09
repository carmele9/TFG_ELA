from modelos.HDBSCANModel import HDBSCANModel
from modelos.DBSCANModel import DBSCANModel
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


##### Test para el modelo DBSCAN #####
@pytest.fixture
# Datos sintéticos para DBSCAN
def synthetic_df():
    # Creamos datos sintéticos con 3 clusters
    X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return df


## Test para verificar el pipeline completo de DBSCAN
def test_dbscan_pipeline(synthetic_df, monkeypatch):
    # Evitamos que los métodos que grafican muestren ventanas
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    model = DBSCANModel(synthetic_df, min_samples=3)

    # Preparamos datos con PCA a 2 componentes
    df_pca = model.prepare_data_for_clustering(n_components=2)
    assert df_pca.shape[1] == 2

    # Forzamos un epsilon fijo para que el clustering sea reproducible
    clustered_df = model.fit(epsilon=1.0, auto_eps=False)
    assert "cluster" in clustered_df.columns
    assert model.labels_ is not None
    assert len(model.labels_) == len(synthetic_df)

    # Evaluamos silhouette score (debe ser un float entre -1 y 1)
    score = model.evaluate()
    assert isinstance(score, float)
    assert -1 <= score <= 1

    # Probamos plot_clusters (sin que bloquee la ejecución)
    model.plot_clusters("PC1", "PC2")


##### Test para el modelo HDBSCAN #####
@pytest.fixture
# Datos sintéticos para HDBSCAN
def synthetic_df_hdbscan():
    # Datos sintéticos con 3 clusters bien separados
    X, _ = make_blobs(n_samples=120, centers=3, n_features=6, cluster_std=0.5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    # Añadimos columna _etiqueta para comprobar que se elimina
    df["dummy_etiqueta"] = np.random.randint(0, 2, size=len(df))
    return df

#Test para verificar el pipeline completo de HDBSCAN
def test_hdbscan_pipeline(synthetic_df_hdbscan, monkeypatch):
    # Evitar que los plots bloqueen
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    model = HDBSCANModel(synthetic_df_hdbscan, min_cluster_size=5)

    # 1. Preparar datos con PCA a 2 componentes
    df_pca = model.prepare_data_for_clustering(n_components=2)
    assert df_pca.shape[1] == 2
    # Comprobar que se eliminó la columna _etiqueta
    assert not any(col.endswith("_etiqueta") for col in model.df.columns)

    # 2. Ajustar el modelo
    clustered_df = model.fit()
    assert "cluster" in clustered_df.columns
    assert len(clustered_df) == len(synthetic_df_hdbscan)
    assert model.labels_ is not None

    # 3. Evaluar clusters
    score = model.evaluate()
    if score is not None:  # Puede ser None si detecta solo ruido
        assert isinstance(score, float)
        assert -1 <= score <= 1

    # 4. Probar visualizaciones
    model.plot_clusters("PC1", "PC2")
    model.plot_outlier_scores()