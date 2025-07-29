import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, \
    auc
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import L2
from sklearn.preprocessing import MinMaxScaler


class AutoencoderModel:
    """
    Modelo de Autoencoder para detección de anomalías.
    - Entrena una red neuronal para reconstruir entradas normales.
    - Detecta anomalías cuando el error de reconstrucción es alto.
    - Evalúa con métricas como Precision, Recall, F1-score y AUC.
    """

    def __init__(self, df, encoding_dim=16, learning_rate=0.001):
        """
        Args:
            df (pd.DataFrame): Dataset de entrada (ya preprocesado y escalado).
            encoding_dim (int): Tamaño de la capa latente.
            learning_rate (float): Tasa de aprendizaje.
        """

        # Filtrar y escalar solo las columnas relevantes que estén en el DataFrame
        self.prepare_data(df)

        # Hiperparámetros y configuración del modelo
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate

        # Inicialización de variables del modelo
        self.model = None
        self.history = None
        self.reconstruction_errors = None

        print(f"Columnas usadas para el autoencoder: {self.features}")

    def prepare_data(self, df):
        """
        Prepara el dataset para el Autoencoder:
        - Escala las variables fisiológicas relevantes con MinMaxScaler.
        - Mantiene sin escalar las columnas de eventos.
        - Genera etiquetas binarias de anomalía según eventos clínicos clave.

        Args:
            df (pd.DataFrame): Dataset sin escalar, con columnas fisiológicas y eventos.
        """

        # Features fisiológicas que usa tu modelo, solo si existen en df
        features_fisio = [
            'aceleracion_x', 'aceleracion_y', 'aceleracion_z',
            'magnitud_movimiento', 'actividad_estimada',
            'spo2', 'frecuencia_cardiaca', 'pulsatility_index',
            'senal_respiratoria', 'frecuencia_respiratoria',
            'variabilidad_respiratoria', 'amplitud_instante',
            'senal_sueno'
        ]
        features_fisio = [f for f in features_fisio if f in df.columns]

        # Columnas de eventos relevantes
        eventos_cols = [
            'hipoxia_sostenida',
            'hipovent_sostenido',
            'inmovilidad_sostenida',
            'frag_sueno_sostenido',
            'empeoramiento'
        ]
        eventos_cols = [col for col in eventos_cols if col in df.columns]
        df_fisio = df[features_fisio].copy()

        # Escalar features fisiológicas
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_fisio)
        df_scaled_features = pd.DataFrame(scaled_features, columns=features_fisio, index=df.index)

        # Concatenar con eventos sin escalar
        df_prepared = pd.concat([df_scaled_features, df[eventos_cols]], axis=1)

        # Generar etiquetas binarias: 1 si algún evento > 0, 0 en otro caso
        y_true = (df[eventos_cols].sum(axis=1) > 0).astype(int).values

        # Guardar atributos en la instancia
        self.df = df_prepared
        self.X = df_scaled_features.values  # para el entrenamiento (solo características fisiológicas escaladas)
        self.y_true = y_true
        self.features = features_fisio

        print(
            f"Preparado dataset con {df_prepared.shape[1]} columnas: {len(features_fisio)} fisiológicas escaladas + {len(eventos_cols)} eventos sin escala.")
        print(f"Etiquetas generadas con {y_true.sum()} anomalías (1) y {len(y_true) - y_true.sum()} normales (0).")

    def build_model(self):
        """Construye un modelo de autoencoder optimizado."""
        input_dim = self.X.shape[1]
        input_layer = Input(shape=(input_dim,))

        # Encoder
        encoded = Dense(64, activation='relu', kernel_regularizer=L2(1e-5))(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu', kernel_regularizer=L2(1e-5))(encoded)
        bottleneck = Dense(self.encoding_dim, activation='relu', kernel_regularizer=L2(1e-5))(encoded)

        # Decoder
        decoded = Dense(32, activation='relu', kernel_regularizer=L2(1e-5))(bottleneck)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(64, activation='relu', kernel_regularizer=L2(1e-5))(decoded)
        output_layer = Dense(input_dim, activation='sigmoid')(decoded)

        # Compilación
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        print(self.model.summary())

    def train(self, epochs=50, batch_size=256, validation_split=0.1):
        """Entrena el autoencoder. Utiliza EarlyStopping para evitar sobreajuste.
        Args:
            epochs (int): Número de épocas para el entrenamiento.
            batch_size (int): Tamaño del lote.
            validation_split (float): Porcentaje de datos para validación.
        """
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.history = self.model.fit(
            self.X, self.X,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=[early_stop]
        )
        self.plot_training()

    def save_model(self, path="autoencoder_model.h5"):
        """
        Guarda el modelo de autoencoder entrenado en un archivo .h5.
        Args:
            path (str): Ruta donde guardar el modelo.
        """
        if self.model is None:
            print("No hay modelo entrenado para guardar.")
        else:
            self.model.save(path)
            print(f"Modelo guardado en: {path}")

    def load_model(self, path="autoencoder_model.h5"):
        """
        Carga un modelo de autoencoder desde un archivo.h5.
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

    def optimize_threshold(self, true_labels, metric="f1", thresholds=100):
        """
        Encuentra el umbral óptimo para las anomalías maximizando una métrica (por defecto F1-score).
        Args:
            true_labels (array-like): Etiquetas reales (0 = normal, 1 = anomalía).
            metric (str): Métrica a optimizar: 'f1', 'precision' o 'recall'.
            thresholds (int): Número de puntos de umbral a evaluar entre min y max de errores.
        """
        if self.reconstruction_errors is None:
            # Si aún no se han calculado los errores, calcularlos primero
            self.detect_anomalies(threshold=None)

        errors = self.reconstruction_errors
        min_err, max_err = errors.min(), errors.max()
        candidate_thresholds = np.linspace(min_err, max_err, thresholds)

        best_threshold = None
        best_score = -1

        for th in candidate_thresholds:
            preds = (errors > th).astype(int)
            if metric == "f1":
                score = f1_score(true_labels, preds, zero_division=0)
            elif metric == "precision":
                score = precision_score(true_labels, preds, zero_division=0)
            elif metric == "recall":
                score = recall_score(true_labels, preds, zero_division=0)
            else:
                raise ValueError("Métrica no soportada. Usa 'f1', 'precision' o 'recall'.")

            if score > best_score:
                best_score = score
                best_threshold = th

        return best_threshold, best_score

    def detect_anomalies(self, threshold=None):
        """
        Detecta anomalías basándonos en el error de reconstrucción.
        Calcula el error de reconstrucción y marca las anomalías según un umbral.
        Si el umbral es None, se usa el percentil indicado para definirlo automáticamente.
        Args:
            threshold (float): Umbral de error. Si None, se usa el percentil indicado.
        """
        reconstructions = self.model.predict(self.X)
        errors = np.mean((self.X - reconstructions) ** 2, axis=1)
        self.reconstruction_errors = errors

        if threshold is None:
            threshold = np.percentile(errors, 95)
            print(f"Umbral automático de anomalía: {threshold:.6f}")

        anomaly_flags = (errors > threshold).astype(int)
        results = self.df.copy()
        results['reconstruction_error'] = errors
        results['anomaly'] = anomaly_flags
        return results, threshold

    def evaluate(self, true_labels, threshold=None, optimize_metric=None, plot_curves=True):
        """
        Evalúa el modelo con métricas de clasificación de anomalías.
        Usa threshold fijo o el percentil 95 si no se da threshold.

        Args:
            true_labels (array-like): Etiquetas reales (0 = normal, 1 = anomalía).
            threshold (float, optional): Umbral para clasificar anomalías.
            optimize_metric (str, optional): Si se pasa, optimiza el umbral con F1/precision/recall.
            plot_curves (bool): Si True, dibuja curvas ROC y PR.
        """
        if self.reconstruction_errors is None:
            self.detect_anomalies()

        # Umbral
        if threshold is None:
            if optimize_metric is not None:
                threshold, _ = self.optimize_threshold(true_labels, metric=optimize_metric)
                print(f"Usando umbral optimizado ({optimize_metric.upper()}): {threshold:.6f}")
            else:
                threshold = np.percentile(self.reconstruction_errors, 95)
                print(f"Usando umbral por percentil 95: {threshold:.6f}")

        # Predicciones
        results, _ = self.detect_anomalies(threshold=threshold)
        y_pred = results['anomaly'].values
        errors = results['reconstruction_error'].values

        # Métricas
        precision = precision_score(true_labels, y_pred, zero_division=0)
        recall = recall_score(true_labels, y_pred, zero_division=0)
        f1 = f1_score(true_labels, y_pred, zero_division=0)

        # AUC
        try:
            auc_score = roc_auc_score(true_labels, errors)
        except ValueError:
            auc_score = None

        anomalies_detected = int(y_pred.sum())
        total_samples = len(y_pred)
        percent_anomalies = 100 * anomalies_detected / total_samples

        # Resumen
        print("\n=== RESUMEN DE EVALUACIÓN ===")
        print(f"Total de muestras: {total_samples}")
        print(f"Anomalías detectadas: {anomalies_detected} ({percent_anomalies:.2f}%)")
        print(f"Umbral usado: {threshold:.6f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"AUC:       {auc_score if auc_score is not None else 'No definido'}")
        print("=============================\n")

        if plot_curves and auc_score is not None:
            self.plot_roc_pr_curves(true_labels, errors)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc_score,
            "threshold": threshold,
            "anomalies_detected": anomalies_detected
        }

    def plot_roc_pr_curves(self, true_labels, scores):
        """Genera curvas ROC y Precision-Recall."""
        plt.figure(figsize=(12, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(true_labels, scores)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
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
