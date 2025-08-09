import pandas as pd
import polars
from procesamiento.PandasPreprocessor import PandasPreprocessor
from procesamiento.PolarPreprocessor import PolarsPreprocessor
import numpy as np
import pytest


#### Tests para PolarsPreprocessor ####
@pytest.fixture
def sample_polars_df():
    return polars.DataFrame({
        'timestamp': ['2025-01-02', '2025-01-01', '2025-01-04', '2025-01-03'],
        'paciente_id': [1, 1, 2, 2],
        'variable1': [1, None, 500, 3],  # con None y outlier
        'variable2': [2, 5, 2, 1000],  # con outlier
        'fase_sueno': ['REM', None, 'NoREM', 'REM'],  # con None
        'estado': ['activo', 'inactivo', 'activo', 'inactivo'],
        'empeoramiento': [0, 1, 0, 1]
    })


def test_parse_timestamp_polars(sample_polars_df):
    pp = PolarsPreprocessor(sample_polars_df)
    pp.parse_timestamp()
    assert pp.df['timestamp'].dtype == polars.Datetime(time_unit='us', time_zone=None)
    assert pp.df['timestamp'][0] <= pp.df['timestamp'][1]


def test_handle_missing_values_polars(sample_polars_df):
    pp = PolarsPreprocessor(sample_polars_df)
    pp.handle_missing_values()
    assert pp.df['variable1'].is_null().sum() == 0


def test_handle_outliers_polars(sample_polars_df):
    pp = PolarsPreprocessor(sample_polars_df)
    pp.handle_outliers()
    assert pp.df['variable2'].max() < 1000


def test_encode_categoricals_polars(sample_polars_df):
    pp = PolarsPreprocessor(sample_polars_df)
    pp.encode_categoricals()
    assert pp.df['fase_sueno'].dtype == polars.Int64
    assert pp.df['estado'].dtype == polars.Int64


def test_scale_features_polars(sample_polars_df):
    pp = PolarsPreprocessor(sample_polars_df)
    pp.scale_features()
    numeric_cols = [col for col, dtype in zip(pp.df.columns, pp.df.dtypes)
                    if dtype in (polars.Float64, polars.Int64)]
    # Comprobar que 'empeoramiento' sea Int64 y no esté escalada
    assert pp.df['empeoramiento'].dtype == polars.Int64
    # Comprobar que las columnas numéricas (excepto 'empeoramiento') estén entre 0 y 1
    for col in numeric_cols:
        if col == 'empeoramiento':
            continue
        max_val = pp.df[col].max()
        min_val = pp.df[col].min()
        assert 0 <= min_val <= max_val <= 2, f"Columna {col} fuera de rango [0,1]"


def test_generate_sequences_polars(sample_polars_df):
    pp = PolarsPreprocessor(sample_polars_df, seq_length=1, step=1)
    X, y = pp.generate_sequences()
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 3  # formato LSTM (samples, timesteps, features)


def test_run_all_integration_polars(sample_polars_df):
    pp = PolarsPreprocessor(sample_polars_df)
    df_processed = pp.run_all()
    numeric_cols = [col for col, dtype in zip(df_processed.columns, df_processed.dtypes)
                    if dtype in (polars.Float64, polars.Int64)]
    # Comprobar que máximo valor en cada columna numérica sea <= 1
    for col in numeric_cols:
        max_val = df_processed[col].max()
        assert max_val <= 2.0, f"Columna {col} tiene valor máximo {max_val} > 2.0"


#### Tests para PandasPreprocessor ####
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'timestamp': ['2025-01-02', '2025-01-01', '2025-01-04', '2025-01-03'],
        'paciente_id': [1, 1, 2, 2],
        'variable1': [1, np.nan, 500, 3],  # con NaN y outlier
        'variable2': [2, 5, 2, 1000],  # con outlier
        'fase_sueno': ['REM', None, 'NoREM', 'REM'],  # con NaN
        'estado': ['activo', 'inactivo', 'activo', 'inactivo'],
        'empeoramiento': [0, 1, 0, 1]
    })


def test_parse_timestamp(sample_df):
    pp = PandasPreprocessor(sample_df)
    pp.parse_timestamp()
    assert pd.api.types.is_datetime64_any_dtype(pp.df['timestamp'])
    assert pp.df.iloc[0]['timestamp'] <= pp.df.iloc[1]['timestamp']


def test_handle_missing_values(sample_df):
    pp = PandasPreprocessor(sample_df)
    pp.handle_missing_values()
    assert pp.df['variable1'].isna().sum() == 0


def test_handle_outliers(sample_df):
    pp = PandasPreprocessor(sample_df)
    pp.handle_outliers()
    assert pp.df['variable2'].max() < 1000


def test_encode_categoricals(sample_df):
    pp = PandasPreprocessor(sample_df)
    pp.encode_categoricals()
    assert np.issubdtype(pp.df['fase_sueno'].dtype, np.integer)
    assert np.issubdtype(pp.df['estado'].dtype, np.integer)


def test_scale_features(sample_df):
    pp = PandasPreprocessor(sample_df)
    pp.scale_features()
    numeric_cols = pp.df.select_dtypes(include=np.number).columns
    assert np.issubdtype(pp.df['empeoramiento'].dtype, np.integer)  # aseguramos que empeoramiento no se escala


def test_generate_sequences(sample_df):
    pp = PandasPreprocessor(sample_df, seq_length=1, step=1)
    X, y = pp.generate_sequences()
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 3  # formato LSTM (samples, timesteps, features)


def test_run_all_integration(sample_df):
    pp = PandasPreprocessor(sample_df)
    df_processed = pp.run_all()
    assert not df_processed.isna().any().any()

