import numpy as np
import pandas as pd
from datetime import datetime


class SensorRespiracion:
    def __init__(self, paciente_id, fase_ela=1, duration=300, sampling_rate=1, offset=0.0, variance=0.3):
        """
        Simula una señal de respiración periódica con ruido gaussiano.

        Args:
            paciente_id (str): ID del paciente.
            fase_ela (int): Fase clínica (1, 2, 3).
            duration (int): Duración de la simulación en segundos.
            sampling_rate (int): Frecuencia de muestreo en Hz.
            offset (float): Desplazamiento vertical de la señal.
            variance (float): Intensidad del ruido gaussiano.
        """
        self.paciente_id = paciente_id
        self.fase_ela = fase_ela
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.offset = offset
        self.variance = variance
        # Tiempo base para sincronización
        self.time = np.linspace(0, duration, int(duration * sampling_rate))
        self.timestamps = pd.date_range(datetime.now(), periods=len(self.time), freq=f"{int(1000 / sampling_rate)}ms")

    def simular(self):
        """
        Genera una señal simulada de respiración con ruido.
        Devuelve un DataFrame con variables fisiológicas realistas.
        """

        # 1. Parámetros según fase ELA
        if self.fase_ela == 1:
            frecuencia_resp = 0.25  # 15 rpm (normal)
            amplitud = 2.0
        elif self.fase_ela == 2:
            frecuencia_resp = 0.20  # 12 rpm
            amplitud = 1.5
        else:  # Fase 3
            frecuencia_resp = 0.15  # 9 rpm (hipoventilación)
            amplitud = 1.0

        # 2. Señal respiratoria sinusoidal
        senal_base = amplitud * np.sin(2 * np.pi * frecuencia_resp * self.time)

        # 3. Ruido gaussiano
        ruido = np.random.normal(0, self.variance, size=len(self.time))

        # 4. Señal final
        senal_resp = senal_base + ruido + self.offset

        # 5. Derivar variables fisiológicas

        # Frecuencia respiratoria estimada (usamos constante porque la señal es simulada)
        frecuencia_rpm = np.full(len(senal_resp), frecuencia_resp * 60)

        # Variabilidad respiratoria (simula cambios entre ciclos)
        variabilidad = 0.1 * np.random.randn(len(senal_resp))

        # Amplitud respiratoria simulada (puede disminuir con ELA)
        amplitud_instante = np.abs(senal_resp)

        # Indicador de hipoventilación: respiración superficial y lenta
        evento_hipoventilacion = ((frecuencia_rpm < 10) & (amplitud_instante < 0.8)).astype(int)

        # Respiración superficial prolongada (amplitud reducida < 0.3)
        evento_brp = (np.abs(senal_resp) < 0.3).astype(int)

        # 6. Crear DataFrame
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "paciente_id": self.paciente_id,
            "fase_ela": self.fase_ela,
            "senal_respiratoria": senal_resp,
            "frecuencia_respiratoria": frecuencia_rpm,
            "variabilidad_respiratoria": variabilidad,
            "amplitud_instante": amplitud_instante,
            "evento_hipoventilacion": evento_hipoventilacion,
            "evento_brp": evento_brp
        })

        return df
