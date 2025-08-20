import numpy as np
import pandas as pd
from datetime import datetime


class SensorSpO2:
    def __init__(self, paciente_id, fase_ela=1, duration=300, sampling_rate=1, offset=0.0, variance=0.5):
        """
        Simula una señal de SpO2 con ruido y deriva de línea base.
        También incluye frecuencia cardíaca asociada.
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
        Genera una señal simulada de SpO2 y frecuencia cardíaca con realismo fisiológico.
        Devuelve un DataFrame con las variables simuladas.
        Returns:
            pd.DataFrame: DataFrame con las columnas:
                - timestamp: Marca de tiempo de la medición.
                - paciente_id: ID del paciente.
                - fase_ela: Fase del ELA (1, 2 o 3).
                - spo2: Nivel de SpO2 simulado.
                - frecuencia_cardiaca: Frecuencia cardíaca simulada.
                - pulsatility_index: Índice de pulsatilidad simulado.
                - evento_hipoxia: Evento de hipoxia (1 si SpO2 < 90%, 0 en caso contrario).
        """
        # 1. Valor base por fase ELA (SpO2)
        if self.fase_ela == 1:
            base_spo2 = 96
            amplitud = 1.5
            fc_base = 70
        elif self.fase_ela == 2:
            base_spo2 = 93
            amplitud = 2.5
            fc_base = 80
        else:  # Fase 3
            base_spo2 = 88
            amplitud = 3.5
            fc_base = 90

        # Señal base SpO2
        spo2_senal = base_spo2 + amplitud * np.sin(2 * np.pi * 1 * self.time)  # 1 Hz -> Frecuencia de pulso
        baseline_wander = 0.5 * np.sin(2 * np.pi * 0.05 * self.time)
        ruido_spo2 = np.random.normal(0, self.variance, size=len(self.time))
        spo2 = spo2_senal + baseline_wander + ruido_spo2 + self.offset
        spo2 = np.clip(spo2, 80, 100)

        # Evento de hipoxia: SpO2 < 90%
        evento_hipoxia = (spo2 < 90).astype(int)

        # Pulsatility Index
        pulsatility_index = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * self.time) + np.random.normal(0, 0.05, size=len(self.time))
        pulsatility_index = np.clip(pulsatility_index, 0.2, 1.2)

        # 9. Frecuencia cardíaca simulada (con realismo y correlación leve con SpO₂)
        fc_variabilidad = 5 * np.sin(2 * np.pi * 0.03 * self.time)  # oscilación lenta ≈ 30s
        ruido_fc = np.random.normal(0, 2, size=len(self.time))

        # Correlación inversa: a menor SpO₂, mayor FC
        # Se calcula en bpm: cada punto de caída en SpO₂ eleva FC en ~0.4 bpm
        correlacion_spo2 = 0.4 * (95 - spo2)  # base en 95%
        frecuencia_cardiaca = fc_base + fc_variabilidad + ruido_fc + correlacion_spo2
        frecuencia_cardiaca = np.clip(frecuencia_cardiaca, 50, 120)

        # 10. DataFrame final
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "paciente_id": self.paciente_id,
            "fase_ela": self.fase_ela,
            "spo2": spo2,
            "frecuencia_cardiaca": frecuencia_cardiaca,
            "pulsatility_index": pulsatility_index,
            "evento_hipoxia": evento_hipoxia
        })

        return df
