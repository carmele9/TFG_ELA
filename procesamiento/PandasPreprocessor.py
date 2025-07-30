import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class PandasPreprocessor:
    """
    Clase de preprocesamiento de datos usando pandas.
    Realiza imputación, manejo de outliers, escalado, codificación categórica,
    y generación de secuencias para datos simulados de sensores en pacientes con ELA.
    """

    def __init__(self, df, seq_length=60, step=1):
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

    def handle_outliers(self):
        """
        Detecta y corrige outliers en columnas numéricas usando el rango intercuartílico (IQR).
        Los valores fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR] se recortan al límite permitido.
        Se excluyen columnas de eventos, etiquetas, sostenidas y empeoramiento.
        """
        # Palabras clave a excluir
        exclude_keywords = ['evento', 'sostenida', 'etiqueta', 'empeoramiento']

        # Detectar columnas numéricas
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns

        # Excluir columnas por nombre si contienen alguna keyword
        cols_to_process = [
            col for col in num_cols
            if not any(keyword in col.lower() for keyword in exclude_keywords)
        ]

        for col in cols_to_process:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound,
                                    np.where(self.df[col] > upper_bound, upper_bound, self.df[col]))

    def encode_categoricals(self):
        """Codifica variables categóricas como 'fase_sueno' y 'estado'."""
        if 'fase_sueno' in self.df.columns:
            self.df['fase_sueno'] = self.df['fase_sueno'].fillna('desconocida')
            self.df['fase_sueno'] = self.label_encoder.fit_transform(self.df['fase_sueno'])
        if 'estado' in self.df.columns:
            self.df['estado'] = self.label_encoder.fit_transform(self.df['estado'])

    def scale_features(self):
        """Escala variables numéricas entre 0 y 1."""
        excluidas = ['timestamp', 'paciente_id', 'fase_ela', 'estado', 'empeoramiento']
        cols_escalar = [col for col in self.df.columns if col not in excluidas and self.df[col].dtype != 'object']
        self.df[cols_escalar] = self.scaler.fit_transform(self.df[cols_escalar])

    def generate_sequences(self):
        """Crea secuencias tipo LSTM (X, y) por paciente."""
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
        """
        self.parse_timestamp()
        self.handle_missing_values()
        self.handle_outliers()
        self.encode_categoricals()
        self.scale_features()

        if export_path:
            self.export(export_path)

        if generar_secuencias:
            return self.generate_sequences()

        return self.df
