import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class PandasPreprocessor:
    """
    Clase de preprocesamiento de datos usando pandas.
    Realiza imputación, escalado, codificación categórica, y generación de secuencias.
    Pensada para datos simulados de sensores en pacientes con ELA.
    """

    def __init__(self, df, seq_length=60, step=1):
        """
        Inicializa con el DataFrame y parámetros para series temporales.

        Args:
            df (pd.DataFrame): DataFrame de entrada.
            seq_length (int): Longitud de la ventana para LSTM.
            step (int): Paso entre secuencias.
        """
        self.df = df.copy()
        self.seq_length = seq_length
        self.step = step
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def parse_timestamp(self):
        """Convierte la columna de timestamp a datetime y ordena por paciente."""
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['paciente_id', 'timestamp'])

    def handle_missing_values(self):
        """Imputa valores numéricos con la media por columna."""
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy='mean')
        self.df[num_cols] = imputer.fit_transform(self.df[num_cols])

    def encode_categoricals(self):
        """Codifica variables categóricas como 'fase_sueno' y 'estado'."""
        if 'fase_sueno' in self.df.columns:
            self.df['fase_sueno'] = self.df['fase_sueno'].fillna('desconocida')
            self.df['fase_sueno'] = self.label_encoder.fit_transform(self.df['fase_sueno'])
        if 'estado' in self.df.columns:
            self.df['estado'] = self.label_encoder.fit_transform(self.df['estado'])

    def scale_features(self):
        """
        Escala variables numéricas entre 0 y 1.
        No se escalan columnas categóricas ni identificadores.
        """
        excluidas = ['timestamp', 'paciente_id', 'fase_ela', 'estado', 'empeoramiento']
        cols_escalar = [col for col in self.df.columns if col not in excluidas and self.df[col].dtype != 'object']
        self.df[cols_escalar] = self.scaler.fit_transform(self.df[cols_escalar])

    def generate_sequences(self):
        """
        Crea secuencias tipo LSTM (X, y) por paciente. y = etiqueta de empeoramiento.

        Returns:
            X (np.array): Secuencias [n_seqs, seq_length, n_features]
            y (np.array): Etiquetas [n_seqs,]
        """
        features = [col for col in self.df.columns if col not in ['timestamp', 'paciente_id', 'empeoramiento']]
        X, y = [], []

        for paciente_id, group in self.df.groupby('paciente_id'):
            group = group.reset_index(drop=True)
            for i in range(0, len(group) - self.seq_length, self.step):
                seq_x = group.loc[i:i+self.seq_length-1, features].values
                seq_y = group.loc[i+self.seq_length-1, 'empeoramiento']
                X.append(seq_x)
                y.append(seq_y)

        return np.array(X), np.array(y)

    def export(self, path="data_simulada/preprocesado.csv"):
        """Guarda el DataFrame preprocesado."""
        self.df.to_csv(path, index=False)

    def run_all(self, export_path=None, generar_secuencias=False):
        """
        Ejecuta el pipeline de preprocesamiento.

        Args:
            export_path (str): Ruta opcional para guardar el CSV.
            generar_secuencias (bool): Si True, devuelve X, y para LSTM.

        Returns:
            pd.DataFrame o (np.array, np.array): DataFrame preprocesado o secuencias.
        """
        self.parse_timestamp()
        self.handle_missing_values()
        self.encode_categoricals()
        self.scale_features()

        if export_path:
            self.export(export_path)

        if generar_secuencias:
            return self.generate_sequences()

        return self.df