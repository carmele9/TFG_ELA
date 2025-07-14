import numpy as np
import pandas as pd
from datetime import datetime


class SensorIMU:
    def __init__(self, paciente_id, fase_ela=1, duration=300, sampling_rate=10, offset=0.0, variance=0.05, activity_period=(0.3, 0.7)):
        """
        Simula datos de acelerómetro en 3 ejes (X, Y, Z) con patrones de caminar/reposo.

        Args:
            paciente_id (str): ID del paciente.
            fase_ela (int): Fase clínica (1, 2, 3).
            duration (int): Duración de la simulación en segundos.
            sampling_rate (int): Muestras por segundo (Hz).
            offset (float): Desplazamiento aplicado a los ejes.
            variance (float): Intensidad del ruido gaussiano.
            activity_period (tuple): Inicio y fin del período activo como fracción (0.0–1.0).
        """
        self.paciente_id = paciente_id
        self.fase_ela = fase_ela
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.offset = offset
        self.variance = variance
        self.activity_period = activity_period

        self.time = np.linspace(0, duration, int(duration * sampling_rate))
        self.timestamps = pd.date_range(datetime.now(), periods=len(self.time), freq=f"{int(1000 / sampling_rate)}ms")

    def simular(self):
        """
        Genera señales de aceleración en 3 ejes con actividad simulada y ruido.
        Devuelve un DataFrame con variables derivadas.
        """
        n = len(self.time)
        active = np.full(n, False)

        # Determinar inicio y fin de la actividad (caminar)
        i_start = int(self.activity_period[0] * n)
        i_end = int(self.activity_period[1] * n)
        active[i_start:i_end] = True

        # Eje Y: simula caminar (1.5 Hz) o reposo con ruido leve
        y = np.zeros(n)
        y[active] = 0.5 * np.sin(2 * np.pi * 1.5 * self.time[active])  # pasos
        y[~active] = np.random.normal(0, 0.05, size=np.sum(~active))  # reposo
        y += np.random.normal(0, self.variance, size=n)

        # Eje X: ruido leve centrado en 0
        x = np.random.normal(0, self.variance, size=n)

        # Eje Z: simula la gravedad (1g) + ruido
        z = 1.0 + np.random.normal(0, self.variance, size=n)

        # Magnitud total del movimiento (módulo del vector)
        magnitud_total = np.sqrt(x**2 + y**2 + z**2)

        # Nivel de actividad estimado por derivada (proxy)
        actividad_estimada = np.abs(np.gradient(magnitud_total))

        # Evento de inmovilidad (actividad < umbral)
        evento_inmovilidad = (actividad_estimada < 0.01).astype(int)

        # DataFrame final
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "paciente_id": self.paciente_id,
            "fase_ela": self.fase_ela,
            "aceleracion_x": x,
            "aceleracion_y": y,
            "aceleracion_z": z,
            "magnitud_movimiento": magnitud_total,
            "actividad_estimada": actividad_estimada,
            "evento_inmovilidad": evento_inmovilidad,
            "estado": ["caminar" if a else "reposo" for a in active]
        })

        return df
