import tensorflow as tf

class LSTMModel:
    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(self.output_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32, verbose=1):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, X_input):
        return self.model.predict(X_input)
