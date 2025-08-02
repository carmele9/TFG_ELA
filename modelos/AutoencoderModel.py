import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import seaborn as sns
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import LeakyReLU, BatchNormalization


class AutoencoderModel:
    """
    Modelo de Autoencoder para detección de anomalías.
    - Entrena una red neuronal para reconstruir entradas normales.
    - Detecta anomalías cuando el error de reconstrucción es alto.
    - Evalúa con métricas como Precision, Recall, F1-score y AUC.
    """

    def __init__(self, df, encoding_dim=6, learning_rate=0.001):
        """
        Args:
            df (pd.DataFrame): Dataset de entrada (ya preprocesado y escalado).
            encoding_dim (int): Tamaño de la capa latente.
            learning_rate (float): Tasa de aprendizaje.
        """
        self.df_original = df.copy()

        # Variables fisiológicas relevantes para detectar anomalías
        self.features = [
            'aceleracion_x', 'aceleracion_y', 'aceleracion_z',
            'magnitud_movimiento', 'actividad_estimada',
            'spo2', 'frecuencia_cardiaca', 'pulsatility_index',
            'senal_respiratoria', 'frecuencia_respiratoria',
            'variabilidad_respiratoria', 'amplitud_instante',
            'senal_sueno'
        ]

        # Filtrar solo las columnas relevantes que estén en el DataFrame
        self.features = [col for col in self.features if col in df.columns]
        self.df = df[self.features].copy()
        self.X = self.df.values  # Dataset listo para el modelo

        # Hiperparámetros y configuración del modelo
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate

        # Inicialización de variables del modelo
        self.model = None
        self.history = None
        self.reconstruction_errors = None

        print(f"Columnas usadas para el autoencoder: {self.features}")

    def build_model(self):
        """Construye un modelo de autoencoder optimizado con arquitectura profunda y activaciones LeakyReLU."""
        input_dim = self.X.shape[1]
        input_layer = Input(shape=(input_dim,))

        # Encoder
        encoded = Dense(128, kernel_regularizer=L2(1e-5))(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = LeakyReLU(negative_slope=0.1)(encoded)
        encoded = Dropout(0.2)(encoded)

        encoded = Dense(64, kernel_regularizer=L2(1e-5))(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = LeakyReLU(negative_slope=0.1)(encoded)

        encoded = Dense(32, kernel_regularizer=L2(1e-5))(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = LeakyReLU(negative_slope=0.1)(encoded)

        bottleneck = Dense(self.encoding_dim, kernel_regularizer=L2(1e-5))(encoded)
        bottleneck = BatchNormalization()(bottleneck)
        bottleneck = LeakyReLU(negative_slope=0.1)(bottleneck)
        bottleneck = Dropout(0.2)(bottleneck)

        # Decoder (espejo del encoder)
        decoded = Dense(32, kernel_regularizer=L2(1e-5))(bottleneck)
        decoded = BatchNormalization()(decoded)
        decoded = LeakyReLU(negative_slope=0.1)(decoded)
        decoded = Dropout(0.2)(decoded)

        decoded = Dense(64, kernel_regularizer=L2(1e-5))(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = LeakyReLU(negative_slope=0.1)(decoded)
        decoded = Dropout(0.2)(decoded)

        decoded = Dense(128, kernel_regularizer=L2(1e-5))(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = LeakyReLU(negative_slope=0.1)(decoded)

        # Salida
        output_layer = Dense(input_dim, activation='linear')(decoded)

        # Compilación
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=Huber(delta=1.0))
        print(self.model.summary())

    def train(self, epochs=50, batch_size=256, validation_split=0.1):
        """Entrena el autoencoder. Utiliza EarlyStopping para evitar sobreajuste."""
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1, min_delta=1e-4)
        self.history = self.model.fit(
            self.X, self.X,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=[early_stop]
        )
        self.plot_training()

    def save_model(self, path="autoencoder_model.keras"):
        """
        Guarda el modelo de autoencoder entrenado en el formato nativo de Keras (.keras).

        Args:
            path (str): Ruta donde guardar el modelo.
        """
        if self.model is None:
            print("No hay modelo entrenado para guardar.")
        else:
            self.model.save(path)  # Formato .keras detectado automáticamente
            print(f"Modelo guardado en: {path}")

    def load_model(self, path="autoencoder_model.keras"):
        """
        Carga un modelo de autoencoder desde un archivo.keras.
        Args:
            path (str): Ruta del archivo.h5 a cargar.
        """
        self.model = load_model(path)
        print(f"Modelo cargado desde: {path}")

    def plot_training(self):
        """Grafica la curva de pérdida de entrenamiento y validación."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.history.history['loss'], label='Training Loss', color='blue')
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Curva de pérdida del Autoencoder')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def detect_anomalies(self, threshold=None, percentile=90):
        """
        Detecta anomalías basándonos en el error de reconstrucción.
        Calcula el error de reconstrucción y marca las anomalías según un umbral.
        """
        reconstructions = self.model.predict(self.X)
        errors = np.mean((self.X - reconstructions) ** 2, axis=1)
        self.reconstruction_errors = errors

        if threshold is None:
            threshold = np.percentile(errors, percentile)
            print(f"Umbral automático de anomalía: {threshold:.6f}")

        anomaly_flags = (errors > threshold).astype(int)
        results = self.df.copy()
        results['reconstruction_error'] = errors
        results['anomaly'] = anomaly_flags
        return results, threshold

    def optimize_threshold(self, true_labels, plot=True):
        """
        Optimiza el umbral basado en el F1-score máximo usando los errores de reconstrucción.
        Args:
            true_labels (array-like): Etiquetas reales (0 = normal, 1 = anomalía).
            plot (bool): Sí se desea mostrar la curva precision/recall/F1 vs. threshold.
        """
        if self.reconstruction_errors is None:
            reconstructions = self.model.predict(self.X)
            self.reconstruction_errors = np.mean((self.X - reconstructions) ** 2, axis=1)

        precisions, recalls, thresholds = precision_recall_curve(true_labels, self.reconstruction_errors)
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precisions[:-1], label='Precision')
            plt.plot(thresholds, recalls[:-1], label='Recall')
            plt.plot(thresholds, f1_scores[:-1], label='F1 Score', linestyle='--')
            plt.axvline(x=best_threshold, color='red', linestyle=':', label=f'Mejor umbral: {best_threshold:.6f}')
            plt.xlabel('Umbral de reconstrucción')
            plt.ylabel('Score')
            plt.title('Optimización de umbral según F1')
            plt.legend()
            plt.grid(True)
            plt.show()

            print(f"Mejor F1: {best_f1:.4f} con umbral: {best_threshold:.6f}")

        return best_threshold

    def evaluate(self, true_labels, threshold=None, optimize=True):
        """
        Evalúa el modelo con métricas de clasificación de anomalías.
        Args:
            true_labels (array-like): Etiquetas reales (0 = normal, 1 = anomalía).
            threshold (float or None): Umbral para clasificar anomalías. Si None y optimize=True, se busca el mejor.
            optimize (bool): Si es True, busca el umbral que maximiza el F1.
        """
        if threshold is None and optimize:
            threshold = self.optimize_threshold(true_labels, plot=True)

        results, threshold = self.detect_anomalies(threshold)
        y_pred = results['anomaly'].values

        precision = precision_score(true_labels, y_pred, zero_division=0)
        recall = recall_score(true_labels, y_pred, zero_division=0)
        f1 = f1_score(true_labels, y_pred, zero_division=0)
        auc = roc_auc_score(true_labels, results['reconstruction_error'])

        print("Métricas de evaluación:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        return {"precision": precision, "recall": recall, "f1": f1, "auc": auc}

    def plot_confusion_matrix(self, true_labels):
        """
        Grafica la matriz de confusión para evaluar el rendimiento del modelo.
        Args:
            true_labels (array-like): Etiquetas reales (0 = normal, 1 = anomalía).
        """
        results, _ = self.detect_anomalies()
        y_pred = results['anomaly'].values

        cm = confusion_matrix(true_labels, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Normal', 'Anomalía'], yticklabels=['Normal', 'Anomalía'])
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Realidad')
        plt.show()

    def plot_error_distribution(self):
        """Grafica la distribución de errores de reconstrucción."""
        if self.reconstruction_errors is None:
            print("Primero ejecuta detect_anomalies().")
            return
        plt.figure(figsize=(8, 5))
        sns.histplot(self.reconstruction_errors, bins=50, kde=True)
        plt.title("Distribución de errores de reconstrucción")
        plt.xlabel("Error")
        plt.ylabel("Frecuencia")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
