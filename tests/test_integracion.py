from sklearn.model_selection import train_test_split
from core.SimularDataset import SimuladorDataset
from procesamiento.PandasPreprocessor import PandasPreprocessor
from modelos.LSTMModel import LSTMModel


def test_pipeline_completo(tmp_path):
    sim = SimuladorDataset(paciente_id="test_paciente")
    df = sim.generar()
    proc = PandasPreprocessor(df)
    X, y = proc.run_all(df, generar_secuencias=True)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (X.shape[1], X.shape[2])
    model = LSTMModel(input_shape=input_shape)
    model.build_model(input_shape=input_shape, lstm_units_1=64, lstm_units_2=32, dense_units=16, dropout_rate=0.2, learning_rate=0.001)
    model.train(X_train, y_train, epochs=1, batch_size=2)
    preds = model.predict(y_test)
    assert preds is not None
    assert preds.shape[0] == X.shape[0]
