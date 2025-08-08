import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modelos.LSTMModel import LSTMModel
from modelos.AutoencoderModel import AutoencoderModel
from modelos.HDBSCANModel import HDBSCANModel
from modelos.DBSCANModel import DBSCANModel


def test_lstm_model():
    input_shape = (5, 3)  # 5 pasos de tiempo, 3 características
    model = LSTMModel(input_shape=input_shape, lstm_units_1=64, lstm_units_2=32, dense_units=64, dropout_rate=0.2, learning_rate=0.001)
    model.build_model(input_shape, lstm_units_1=64, lstm_units_2=32, dense_units=64, dropout_rate=0.2, learning_rate=0.001)
    X = np.random.rand(10, 5, 3)  # 10 muestras, 5 pasos de tiempo, 3 características
    y = np.random.randint(0, 2, (10,))  # Etiquetas binarias
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.train(X_train, y_train, epochs=1, batch_size=2)
    preds = model.predict(y_test)
    assert preds.shape[0] == X_test.shape[0]  # Verifica que las predicciones tengan el mismo número de muestras que X_test
    assert preds.shape[1] == 1  # Verifica que la salida sea de una sola clase
    assert preds.shape == (10, 1)  # Salida de una sola clase


def test_autoencoder_single_class():
    df = pd.DataFrame()
    df['feature1'] = np.random.rand(100)
    df['feature2'] = np.random.rand(100)
    df['feature3'] = np.random.rand(100)
    df['feature4'] = np.random.rand(100)
    df['feature5'] = np.random.rand(100)
    df['feature6'] = np.random.rand(100)
    model = AutoencoderModel(df)
    model.build_model()
    X = df.values
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model.train(X, batch_size=10)
    preds = model.evaluate(X)
    assert preds.shape == (100, 1)  # Salida de una sola clase


def test_dbscan_model():
    df = pd.DataFrame()
    df['feature1'] = np.random.rand(100)
    df['feature2'] = np.random.rand(100)
    df['feature3'] = np.random.rand(100)
    df['feature4'] = np.random.rand(100)
    df['feature5'] = np.random.rand(100)
    df['feature6'] = np.random.rand(100)
    model = DBSCANModel(df, eps=0.5, min_samples=2)
    model.fit()
    labels = model.evaluate()
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 100


def test_hdbscan_model():
    df = pd.DataFrame()
    df['feature1'] = np.random.rand(100)
    df['feature2'] = np.random.rand(100)
    df['feature3'] = np.random.rand(100)
    df['feature4'] = np.random.rand(100)
    df['feature5'] = np.random.rand(100)
    df['feature6'] = np.random.rand(100)
    model = HDBSCANModel(df, min_cluster_size=2)
    model.fit()
    labels = model.evaluate()
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 100

