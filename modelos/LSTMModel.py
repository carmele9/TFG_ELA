import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.python.keras.models import load_model


class LSTMModel:
    def __init__(self, input_shape, lstm_units_1=64, lstm_units_2=32, dense_units=64, dropout_rate=0.2, learning_rate=0.001):
        """
        Inicializa el modelo LSTM con los parámetros especificados.
        Args:
            input_shape (tuple): Forma de entrada de los datos (n_timesteps, n_features).
            lstm_units_1 (int): Número de unidades en la primera capa LSTM.
            lstm_units_2 (int): Número de unidades en la segunda capa LSTM.
            dense_units (int): Número de unidades en la capa densa.
            dropout_rate (float): Tasa de dropout para regularización.
            learning_rate (float): Tasa de aprendizaje del optimizador Adam.
        """
        self.model = self.build_model(input_shape, lstm_units_1, lstm_units_2, dense_units, dropout_rate, learning_rate)

    def build_model(self, input_shape, lstm_units_1, lstm_units_2, dense_units, dropout_rate, learning_rate):
        """
        Construye el modelo LSTM con las capas especificadas.
        Args:
            input_shape (tuple): Forma de entrada de los datos (n_timesteps, n_features).
            lstm_units_1 (int): Número de unidades en la primera capa LSTM.
            lstm_units_2 (int): Número de unidades en la segunda capa LSTM.
            dense_units (int): Número de unidades en la capa densa.
            dropout_rate (float): Tasa de dropout para regularización.
            learning_rate (float): Tasa de aprendizaje del optimizador Adam.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Bidirectional(LSTM(lstm_units_1, return_sequences=True)))
        model.add(Dropout(dropout_rate))

        model.add(Bidirectional(LSTM(lstm_units_2)))
        model.add(Dense(dense_units))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=20, use_early_stopping=True, patience=5):
        """
        Entrena el modelo LSTM con los datos de entrenamiento y validación.
        Args:
            X_train (np.ndarray): Datos de entrenamiento.
            y_train (np.ndarray): Etiquetas de entrenamiento.
            X_val (np.ndarray, optional): Datos de validación. Si se proporciona, se usará para early stopping.
            y_val (np.ndarray, optional): Etiquetas de validación. Si se proporciona, se usará para early stopping.
            batch_size (int): Tamaño del lote para el entrenamiento.
            epochs (int): Número de épocas para entrenar.
            use_early_stopping (bool): Si True, usa early stopping basado en la pérdida de validación.
            patience (int): Paciencia para early stopping.
        """
        callbacks = []
        if use_early_stopping and X_val is not None and y_val is not None:
            early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
            callbacks.append(early_stop)

        if X_val is not None and y_val is not None:
            history = self.model.fit(X_train, y_train,
                                     validation_data=(X_val, y_val),
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     callbacks=callbacks,
                                     verbose=1)
        else:
            history = self.model.fit(X_train, y_train,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     callbacks=callbacks,
                                     verbose=1)
        return history

    def predict(self, X_test, threshold=0.5):
        """
        Realiza predicciones con el modelo LSTM en los datos de prueba.
        Args:
            X_test (np.ndarray): Datos de prueba.
            threshold (float): Umbral para convertir probabilidades en etiquetas binarias.
        """
        probs = self.model.predict(X_test)
        preds = (probs >= threshold).astype(int)
        return preds, probs

    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evalúa el modelo LSTM en los datos de prueba y muestra métricas de rendimiento.
        Args:
            X_test (np.ndarray): Datos de prueba.
            y_test (np.ndarray): Etiquetas de prueba.
            threshold (float): Umbral para convertir probabilidades en etiquetas binarias.
        """
        preds, probs = self.predict(X_test, threshold)

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = None
            warnings.warn("ROC AUC no se puede calcular: solo hay una clase verdadera en y_test")

        print(f" Accuracy: {acc:.4f}")
        if auc is not None:
            print(f" ROC AUC: {auc:.4f}")
        print("\n Classification Report:")
        print(classification_report(y_test, preds))
        print("\n Confusion Matrix:")
        print(cm)

        if auc is not None:
            fpr, tpr, _ = roc_curve(y_test, probs)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid()
            plt.show()

        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        return {
            'accuracy': acc,
            'roc_auc': auc,
            'report': report,
            'confusion_matrix': cm
        }

    def save_model(self, path="modelos/lstm_model.keras"):
        """
        Guarda el modelo LSTM entrenado en el disco.
        Args:
            path (str): Ruta donde se guardará el modelo.
        """
        if self.model is None:
            print("No hay modelo entrenado para guardar.")
        else:
            self.model.save(path)
            print(f"Modelo guardado en: {path}")

    def load_model(self, path="modelos/lstm_model.keras"):
        """
        Carga un modelo LSTM previamente guardado desde el disco.
        Args:
            path (str): Ruta desde donde se cargará el modelo.
        """
        self.model = load_model(path)
        print(f"Modelo cargado desde: {path}")
