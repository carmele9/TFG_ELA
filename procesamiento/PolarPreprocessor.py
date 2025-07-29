import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class PolarsPreprocessor:
    """
    Preprocesador de datos usando Polars.
    Realiza parsing temporal, imputación, manejo de outliers, codificación, escalado y generación de secuencias.
    Diseñado para datos fisiológicos simulados de pacientes con ELA.
    """

    def __init__(self, df: pl.DataFrame, seq_length: int = 60, step: int = 1):
        self.df = df.clone()
        self.seq_length = seq_length
        self.step = step
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def parse_timestamp(self):
        """Convierte timestamp a datetime y ordena por paciente y tiempo."""
        self.df = self.df.with_columns([
            pl.col("timestamp").str.to_datetime()
        ]).sort(["paciente_id", "timestamp"])

    def handle_missing_values(self):
        """Imputa valores numéricos con la media de cada columna."""
        numeric_cols = [col for col in self.df.columns if self.df[col].dtype in [pl.Float64, pl.Int64]]
        np_array = self.df[numeric_cols].to_numpy()
        imputer = SimpleImputer(strategy="mean")
        np_imputed = imputer.fit_transform(np_array)
        for i, col in enumerate(numeric_cols):
            self.df = self.df.with_columns(pl.Series(name=col, values=np_imputed[:, i]))

    def handle_outliers(self):
        """
        Detecta y ajusta outliers en columnas numéricas usando el rango intercuartílico (IQR).
        Los valores fueron de [Q1 - 1.5*IQR, Q3 + 1.5*IQR] se recortan a dichos límites.
        """
        numeric_cols = [col for col in self.df.columns if self.df[col].dtype in [pl.Float64, pl.Int64]]
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            self.df = self.df.with_columns(
                pl.when(pl.col(col) < lower_bound).then(lower_bound)
                .when(pl.col(col) > upper_bound).then(upper_bound)
                .otherwise(pl.col(col))
                .alias(col)
            )

    def encode_categoricals(self):
        """Codifica variables categóricas como 'fase_sueno' y 'estado'."""
        if 'fase_sueno' in self.df.columns:
            fase = self.df['fase_sueno'].fill_null('desconocida').to_list()
            encoded = self.label_encoder.fit_transform(fase)
            self.df = self.df.with_columns(pl.Series(name='fase_sueno', values=encoded))
        if 'estado' in self.df.columns:
            estado = self.df['estado'].to_list()
            encoded = self.label_encoder.fit_transform(estado)
            self.df = self.df.with_columns(pl.Series(name='estado', values=encoded))

    def scale_features(self):
        """Escala columnas numéricas entre 0 y 1, excluyendo variables categóricas e IDs."""
        excluidas = ['timestamp', 'paciente_id', 'fase_ela', 'estado', 'empeoramiento']
        columnas_escalar = [col for col in self.df.columns if col not in excluidas and self.df[col].dtype in [pl.Float64, pl.Int64]]
        np_vals = self.df[columnas_escalar].to_numpy()
        np_scaled = self.scaler.fit_transform(np_vals)
        for i, col in enumerate(columnas_escalar):
            self.df = self.df.with_columns(pl.Series(name=col, values=np_scaled[:, i]))

    def generate_sequences(self):
        """
        Crea secuencias tipo LSTM (X, y) por paciente. y = etiqueta de empeoramiento.

        Returns:
            X (np.array): Secuencias [n_seqs, seq_length, n_features]
            y (np.array): Etiquetas [n_seqs,]
        """
        features = [col for col in self.df.columns if col not in ['timestamp', 'paciente_id', 'empeoramiento']]
        X, y = [], []

        for paciente_id in self.df['paciente_id'].unique().to_list():
            grupo = self.df.filter(pl.col("paciente_id") == paciente_id).select(features + ['empeoramiento'])
            grupo_np = grupo.to_numpy()
            for i in range(0, len(grupo_np) - self.seq_length, self.step):
                seq_x = grupo_np[i:i+self.seq_length, :-1]
                seq_y = grupo_np[i+self.seq_length-1, -1]
                X.append(seq_x)
                y.append(seq_y)

        return np.array(X), np.array(y)

    def export(self, path="data_simulada/preprocesado_polars.csv"):
        """Guarda el DataFrame preprocesado como CSV."""
        self.df.write_csv(path)

    def run_all(self, export_path=None, generar_secuencias=False, for_autoencoder=False):
        """Ejecuta todo el pipeline.

        Args:
            export_path (str, optional): Ruta para exportar CSV. Si None, no exporta.
            generar_secuencias (bool): Si True, genera y retorna secuencias tipo LSTM.
            for_autoencoder (bool): Si True, solo hace manejo de outliers, missing y categóricas
                                    y exporta con nombre especial.
        """
        self.parse_timestamp()

        if for_autoencoder:
            self.handle_missing_values()
            self.handle_outliers()
            self.encode_categoricals()

            if export_path is None:
                export_path = "data_simulada/preprocesado_polars_autoencoder.csv"
            self.export(export_path)

            # Para autoencoder no generamos secuencias (por defecto)
            return self.df

        # Caso normal completo
        self.handle_missing_values()
        self.handle_outliers()
        self.encode_categoricals()
        self.scale_features()

        if export_path:
            self.export(export_path)

        if generar_secuencias:
            return self.generate_sequences()

        return self.df
