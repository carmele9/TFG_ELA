import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class AutoencoderModel:
    """
    Modelo de Autoencoder para detección de anomalías.
    - Entrena una red neuronal para reconstruir entradas normales.
    - Detecta anomalías cuando el error de reconstrucción es alto.
    - Evalúa con métricas como Precision, Recall, F1-score y AUC.
    """

    def __init__(self, df, encoding_dim=16, learning_rate=0.001, excluded_cols=None):
        """
        Args:
            df (pd.DataFrame): Dataset de entrada.
            encoding_dim (int): Tamaño de la capa latente.
            learning_rate (float): Tasa de aprendizaje.
            excluded_cols (list): Lista opcional de columnas a excluir.
        """
        self.df_original = df.copy()

        # --- 1. Filtrar columnas irrelevantes automáticamente ---
        default_excluded = ['timestamp', 'paciente_id', 'empeoramiento']
        if excluded_cols:
            default_excluded.extend(excluded_cols)

        self.features = [col for col in df.columns
                         if col not in default_excluded and np.issubdtype(df[col].dtype, np.number)]

        self.df = df[self.features].copy()

        # --- 2. Escalado ---
        self.scaler = MinMaxScaler()
        self.X = self.scaler.fit_transform(self.df)

        # --- 3. Hiperparámetros ---
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate

        # --- 4. Modelo ---
        self.autoencoder = None
        self.history = None
        self.reconstruction_errors = None

        print(f"[INFO] Columnas usadas para el autoencoder: {self.features}")

    def build_model(self):
        """Construye el modelo de autoencoder."""
        input_dim = self.X.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(self.encoding_dim, activation="relu")(input_layer)
        decoded = Dense(input_dim, activation="sigmoid")(encoded)
        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        print(self.autoencoder.summary())

    def train(self, epochs=50, batch_size=256, validation_split=0.1):
        """Entrena el autoencoder."""
        self.history = self.autoencoder.fit(
            self.X, self.X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=1
        )

    def detect_anomalies(self, threshold=None, percentile=95):
        """
        Detecta anomalías basándonos en el error de reconstrucción.
        Args:
            threshold (float): Umbral de error. Si None, se usa el percentil indicado.
            percentile (int): Percentil para definir el umbral automáticamente.
        """
        reconstructed = self.autoencoder.predict(self.X)
        self.reconstruction_errors = np.mean((self.X - reconstructed) ** 2, axis=1)

        if threshold is None:
            threshold = np.percentile(self.reconstruction_errors, percentile)
        anomalies = (self.reconstruction_errors > threshold).astype(int)

        results = self.df_original.copy()
        results["reconstruction_error"] = self.reconstruction_errors
        results["anomaly"] = anomalies

        return results, threshold

    def evaluate(self, true_labels, threshold=None):
        """
        Evalúa el modelo con métricas de clasificación.
        Args:
            true_labels (array): Etiquetas reales (0=normal, 1=anómalo).
            threshold (float): Umbral de error para anomalías.
        """
        _, threshold = self.detect_anomalies(threshold)
        pred_labels = (self.reconstruction_errors > threshold).astype(int)

        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        auc = roc_auc_score(true_labels, self.reconstruction_errors)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return {"precision": precision, "recall": recall, "f1": f1, "auc": auc}

    def plot_error_distribution(self):
        """Muestra la distribución del error de reconstrucción."""
        if self.reconstruction_errors is None:
            print("ERROR: Ejecuta detect_anomalies() primero.")
            return
        plt.figure(figsize=(8, 5))
        plt.hist(self.reconstruction_errors, bins=50, alpha=0.7)
        plt.title("Distribución de errores de reconstrucción")
        plt.xlabel("Error")
        plt.ylabel("Frecuencia")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
