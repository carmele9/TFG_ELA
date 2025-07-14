import numpy as np
import pandas as pd
from datetime import datetime


class SensorSueno:
    def __init__(self, paciente_id, fase_ela=1, duration=3600, sampling_rate=1, offset=0.0, variance=0.1):
        """
        Simula fases de sueño en ciclos alternantes (REM, LIGHT, DEEP, AWAKE).

        Args:
            paciente_id (str): ID del paciente.
            fase_ela (int): Fase de la ELA (1, 2, 3).
            duration (int): Duración total en segundos.
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

        self.time = np.linspace(0, duration, int(duration * sampling_rate))
        self.timestamps = pd.date_range(datetime.now(), periods=len(self.time), freq=f"{int(1000 / sampling_rate)}ms")

    def simular(self):
        """
        Genera señales sintéticas que simulan patrones de sueño por fases.
        Devuelve un DataFrame con etiquetas de fase y variables derivadas.
        """

        # Parámetros de fases: freqs y amps simuladas para cada etapa
        fases = ['AWAKE', 'REM', 'LIGHT', 'DEEP']
        freqs = {'AWAKE': 0.3, 'REM': 0.1, 'LIGHT': 0.2, 'DEEP': 0.05}
        amps = {'AWAKE': 0.2, 'REM': 0.5, 'LIGHT': 0.3, 'DEEP': 0.7}

        # Distribución de fases por ciclo de sueño (simplificado)
        if self.fase_ela == 1:
            ciclos = [['LIGHT', 'DEEP', 'REM', 'AWAKE']] * 4
        elif self.fase_ela == 2:
            ciclos = [['LIGHT', 'DEEP', 'REM', 'AWAKE']] * 3 + [['LIGHT', 'REM', 'AWAKE']]
        else:  # Fase 3: sueño más fragmentado
            ciclos = [['LIGHT', 'AWAKE']] * 6

        # Asignar duración de cada segmento
        total_points = len(self.time)
        segmento_pts = total_points // sum(len(c) for c in ciclos)

        senal = np.zeros(total_points)
        etiquetas = np.empty(total_points, dtype=object)
        idx = 0

        for ciclo in ciclos:
            for fase in ciclo:
                seg_len = segmento_pts
                t_local = self.time[idx:idx + seg_len]
                freq = freqs[fase]
                amp = amps[fase]
                sig = amp * np.sin(2 * np.pi * freq * t_local)
                ruido = np.random.normal(0, self.variance, size=seg_len)
                senal[idx:idx + seg_len] = sig + ruido + self.offset
                etiquetas[idx:idx + seg_len] = fase
                idx += seg_len

        # Ajuste final en caso de desajuste
        senal = senal[:total_points]
        etiquetas = etiquetas[:total_points]

        # Detectar evento de fragmentación (cambio frecuente de fase)
        cambios = np.concatenate([np.array([0]), np.diff(pd.Series(etiquetas).factorize()[0])])
        evento_fragmentacion = (np.abs(cambios) > 0).astype(int)

        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "paciente_id": self.paciente_id,
            "fase_ela": self.fase_ela,
            "senal_sueno": senal,
            "fase_sueno": etiquetas,
            "evento_fragmentacion": evento_fragmentacion
        })

        return df
