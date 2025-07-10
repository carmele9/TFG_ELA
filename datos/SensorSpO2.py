import numpy as np
import pandas as pd
from datetime import datetime


class SensorSpO2:
    def __init__(self, paciente_id, fase_ela=1, duration=300, sampling_rate=1, offset=0.0, variance=0.5):
        """
        Simula una señal de SpO2 con ruido y deriva de línea base.

        Args:
            paciente_id (str): ID del paciente.
            fase_ela (int): Fase de la enfermedad (1, 2, 3).
            duration (int): Duración de la simulación en segundos.
            sampling_rate (int): Muestras por segundo (Hz).
            offset (float): Desplazamiento del valor base.
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
        Genera una señal simulada de SpO2 con realismo fisiológico.
        Devuelve un DataFrame con las variables simuladas.
        """
        # 1. Valor base por fase ELA
        if self.fase_ela == 1:
            base_spo2 = 96
            amplitud = 1.5
        elif self.fase_ela == 2:
            base_spo2 = 93
            amplitud = 2.5
        else:  # Fase 3
            base_spo2 = 88
            amplitud = 3.5

        # 2. Señal base oscilante (simula el pulso)
        frecuencia_pulso = 1  # Hz
        spo2_senal = base_spo2 + amplitud * np.sin(2 * np.pi * frecuencia_pulso * self.time)

        # 3. Baseline wander (muy baja frecuencia)
        baseline_wander = 0.5 * np.sin(2 * np.pi * 0.05 * self.time)

        # 4. Ruido gaussiano
        ruido = np.random.normal(0, self.variance, size=len(self.time))

        # 5. Señal final con offset
        spo2 = spo2_senal + baseline_wander + ruido + self.offset
        spo2 = np.clip(spo2, 80, 100)

        # 6. Evento crítico si SpO2 < 90%
        evento_hipoxia = (spo2 < 90).astype(int)

        # 7. Pulsatility Index (PI) simulado (PI alto = mejor perfusión)
        pulsatility_index = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * self.time) + np.random.normal(0, 0.05, size=len(self.time))
        pulsatility_index = np.clip(pulsatility_index, 0.2, 1.2)

        # 8. Construcción del DataFrame
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "paciente_id": self.paciente_id,
            "fase_ela": self.fase_ela,
            "spo2": spo2,
            "pulsatility_index": pulsatility_index,
            "evento_hipoxia": evento_hipoxia
        })

        return df
