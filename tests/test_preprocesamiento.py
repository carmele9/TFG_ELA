import pandas as pd
import polars
from procesamiento.PandasPreprocessor import PandasPreprocessor
from procesamiento.PolarPreprocessor import PolarsPreprocessor


def test_equivalencia_pandas():
    data = {
        "paciente_id": ["PAC_001"]*5,
        "feature1": [1, 2, None, 4, 5],
        "feature2": [5, 4, 3, None, 1]
    }
    df = pd.DataFrame(data)
    pandas_proc = PandasPreprocessor(df)
    df_pandas = pandas_proc.run_all(export_path="tests/data/preprocesado_pandas.csv")


def test_equivalencia_polars():
    data = {
        "paciente_id": ["PAC_001"]*5,
        "feature1": [1, 2, None, 4, 5],
        "feature2": [5, 4, 3, None, 1]
    }
    df = polars.DataFrame(data)
    polars_proc = PolarsPreprocessor(df)
    df_polars = polars_proc.run_all(export_path="tests/data/preprocesado_polars.csv")


def test_equivalencia():
    test_equivalencia_pandas()
    test_equivalencia_polars()
    df_pandas = pd.read_csv("tests/data/preprocesado_pandas.csv")
    df_polars = polars.read_csv("tests/data/preprocesado_polars.csv")
    assert df_pandas.shape == df_polars.shape
    assert set(df_pandas.columns) == set(df_polars.columns)
    assert not df_pandas.isna().any().any()
    assert not df_polars.isna().any().any()
