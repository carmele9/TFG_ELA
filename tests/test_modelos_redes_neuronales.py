import numpy as np
import pandas as pd
from modelos.LSTMModel import LSTMModel
from modelos.AutoencoderModel import AutoencoderModel
import pytest
import tempfile
import os
from tensorflow.keras import backend as K


@pytest.fixture
#### Test para el modelo LSTM ####
def small_dataset():
    np.random.seed(42)
    X = np.random.rand(20, 5, 3)  # 20 muestras, 5 timesteps, 3 features
    y = np.random.randint(0, 2, size=(20,))
    return X, y


@pytest.fixture
# Crear un modelo LSTM con parámetros específicos
def lstm_model():
    return LSTMModel(input_shape=(5, 3), lstm_units_1=4, lstm_units_2=2, dense_units=4)


#Test para verificar la correcta construcción del modelo LSTM
def test_build_model(lstm_model):
    assert lstm_model.model is not None
    assert lstm_model.model.input_shape == (None, 5, 3)


# Test para verificar que el modelo se entrena y predice correctamente
def test_train_and_predict(lstm_model, small_dataset):
    X, y = small_dataset
    history = lstm_model.train(X, y, epochs=1, batch_size=4, use_early_stopping=False)
    assert "loss" in history.history

    preds, probs = lstm_model.predict(X)
    assert preds.shape == (20, 1)
    assert probs.shape == (20, 1)
    assert set(np.unique(preds)).issubset({0, 1})


## Test para verificar que el modelo se entrena y evalúa correctamente
def test_evaluate(lstm_model, small_dataset):
    X, y = small_dataset
    lstm_model.train(X, y, epochs=1, batch_size=4, use_early_stopping=False)
    results = lstm_model.evaluate(X, y)
    assert "accuracy" in results
    assert "roc_auc" in results
    assert "report" in results
    assert "confusion_matrix" in results

    # Limpiar sesión de Keras para evitar problemas en otros tests
    K.clear_session()


#### Test para el modelo Autoencoder ####
@pytest.fixture
def dummy_df():
    np.random.seed(42)
    # Crear DataFrame sintético con las columnas requeridas
    data = {
        'aceleracion_x': np.random.rand(100),
        'aceleracion_y': np.random.rand(100),
        'aceleracion_z': np.random.rand(100),
        'magnitud_movimiento': np.random.rand(100),
        'actividad_estimada': np.random.rand(100),
        'spo2': np.random.rand(100),
        'frecuencia_cardiaca': np.random.rand(100),
        'pulsatility_index': np.random.rand(100),
        'senal_respiratoria': np.random.rand(100),
        'frecuencia_respiratoria': np.random.rand(100),
        'variabilidad_respiratoria': np.random.rand(100),
        'amplitud_instante': np.random.rand(100),
        'senal_sueno': np.random.rand(100)
    }
    return pd.DataFrame(data)


## Test para verificar la correcta construcción y entrenamiento del modelo Autoencoder
def test_autoencoder_pipeline(dummy_df):
    model = AutoencoderModel(dummy_df, encoding_dim=4, learning_rate=0.001)
    model.build_model()
    assert model.model is not None, "El modelo no se construyó correctamente"

    # Entrenamiento rápido para test (pocas épocas)
    model.train(epochs=2, batch_size=16, validation_split=0.2)
    assert model.history is not None, "No se guardó el historial de entrenamiento"

    # Detectar anomalías
    results, threshold = model.detect_anomalies(percentile=80)
    assert 'anomaly' in results.columns
    assert threshold > 0

    # Optimizar umbral
    true_labels = np.random.randint(0, 2, size=len(dummy_df))
    best_threshold = model.optimize_threshold(true_labels, plot=False)
    assert isinstance(best_threshold, float)

    # Evaluar
    metrics = model.evaluate(true_labels, threshold=best_threshold, optimize=False)
    for m in ['precision', 'recall', 'f1', 'auc']:
        assert m in metrics
        assert 0.0 <= metrics[m] <= 1.0
