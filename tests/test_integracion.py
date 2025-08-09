from core.SimularDataset import SimuladorDataset
from procesamiento.PolarPreprocessor import PolarsPreprocessor
from modelos.LSTMModel import LSTMModel
import polars as pl
import pandas as pd
import pytest


# Test de integración del pipeline completo: simulación, preprocesamiento, entrenamiento y evaluación
@pytest.mark.integration
def test_pipeline_integration(tmp_path):
    # Generar dataset sintético
    sim = SimuladorDataset(paciente_id="PAC_TEST", fase_ela=1, duracion=300)
    df_pandas = sim.generar()
    df_pandas["timestamp"] = df_pandas["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    assert not df_pandas.empty, "El simulador no generó datos"

    # Pasar a Polars y preprocesar
    df_polars = pl.from_pandas(df_pandas)
    preproc = PolarsPreprocessor(df_polars)
    X, y = preproc.run_all(generar_secuencias=True)
    assert X is not None, "X no debe ser None"
    assert y is not None, "y no debe ser None"
    assert X.shape[2] == 23, "Cantidad de features no es la esperada"
    assert len(X) == len(y), "X e y no tienen la misma cantidad de muestras"
    assert X.shape[0] == y.shape[0], "Cantidad de muestras en X e y no coincide"

    # Dividir datos en train/val/test (simple para test)
    split_idx = int(0.7 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # Crear y entrenar el modelo
    model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=1, batch_size=8, use_early_stopping=False
    )
    assert history is not None, "El entrenamiento no devolvió historial"

    # Predicciones y evaluación
    preds, probs = model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0], "Cantidad de predicciones incorrecta"

    results = model.evaluate(X_test, y_test)
    assert "accuracy" in results, "Falta métrica de accuracy"
    assert results["accuracy"] >= 0, "Accuracy inválido"
