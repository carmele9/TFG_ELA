import numpy as np
import pandas as pd
from datetime import datetime


class SensorIMU:
    # Diccionario con parámetros por fase clínica
    parametros_por_fase = {
        1: {"activity_period": (0.3, 0.7), "inmovilidad_period": None, "variance": 0.05},
        2: {"activity_period": (0.2, 0.5), "inmovilidad_period": (150, 250), "variance": 0.07},
        3: {"activity_period": (0.1, 0.3), "inmovilidad_period": (100, 400), "variance": 0.1}
    }

    def __init__(self, paciente_id, fase_ela=1, duration=300, sampling_rate=10, offset=0.0,
                 variance=None, activity_period=None, inmovilidad_period=None):
        """
        Simula datos de acelerómetro en 3 ejes (X, Y, Z) con patrones de caminar/reposo.

        Args:
            paciente_id (str): ID del paciente.
            fase_ela (int): Fase clínica (1, 2, 3).
            duration (int): Duración de la simulación en segundos.
            sampling_rate (int): Muestras por segundo (Hz).
            offset (float): Desplazamiento aplicado a los ejes.
            variance (float, opcional): Intensidad del ruido gaussiano.
            activity_period (tuple, opcional): Inicio y fin del período activo como fracción (0.0–1.0).
            inmovilidad_period (tuple o None, opcional): (inicio, fin) en segundos del periodo de inmovilidad sostenida.
        """
        self.paciente_id = paciente_id
        self.fase_ela = fase_ela
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.offset = offset

        # Asignar parámetros por defecto según fase si no se pasan explícitamente
        params = self.parametros_por_fase.get(fase_ela, {})
        self.activity_period = activity_period if activity_period is not None else params.get("activity_period", (0.3, 0.7))
        self.inmovilidad_period = inmovilidad_period if inmovilidad_period is not None else params.get("inmovilidad_period", None)
        self.variance = variance if variance is not None else params.get("variance", 0.05)

        self.time = np.linspace(0, duration, int(duration * sampling_rate))
        self.timestamps = pd.date_range(datetime.now(), periods=len(self.time), freq=f"{int(1000 / sampling_rate)}ms")

    def simular(self):
        # Resto del método igual que antes...

        n = len(self.time)
        active = np.full(n, False)

        i_start = int(self.activity_period[0] * n)
        i_end = int(self.activity_period[1] * n)
        active[i_start:i_end] = True

        y = np.zeros(n)
        y[active] = 0.5 * np.sin(2 * np.pi * 1.5 * self.time[active])  # pasos
        y[~active] = np.random.normal(0, 0.05, size=np.sum(~active))  # reposo
        y += np.random.normal(0, self.variance, size=n)

        x = np.random.normal(0, self.variance, size=n)
        z = 1.0 + np.random.normal(0, self.variance, size=n)

        if self.inmovilidad_period is not None:
            start_idx = int(self.inmovilidad_period[0] * self.sampling_rate)
            end_idx = int(self.inmovilidad_period[1] * self.sampling_rate)
            start_idx = max(0, start_idx)
            end_idx = min(n, end_idx)

            x[start_idx:end_idx] = np.random.normal(0, self.variance / 10, size=(end_idx - start_idx))
            y[start_idx:end_idx] = np.random.normal(0, self.variance / 10, size=(end_idx - start_idx))
            z[start_idx:end_idx] = 1.0 + np.random.normal(0, self.variance / 10, size=(end_idx - start_idx))

        magnitud_total = np.sqrt(x**2 + y**2 + z**2)
        actividad_estimada = np.abs(np.gradient(magnitud_total))
        evento_inmovilidad = (actividad_estimada < 0.01).astype(int)

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


