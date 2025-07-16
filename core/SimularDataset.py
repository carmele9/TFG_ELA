import pandas as pd
from sensores.SensorSpO2 import SensorSpO2
from sensores.SensorRespiracion import SensorRespiracion
from sensores.SensorIMU import SensorIMU
from sensores.SensorSueno import SensorSueno
from etiquetas.EventLabeler import EventLabeler


class SimuladorDataset:
    def __init__(self, paciente_id, fase_ela=1, duracion=600):
        self.paciente_id = paciente_id
        self.fase_ela = fase_ela
        self.duracion = duracion
        self.sampling_rate = {
            "spo2": 1,
            "resp": 1,
            "imu": 10,
            "sueno": 1
        }

    def generar(self):
        # Simulaciones
        df_spo2 = SensorSpO2(self.paciente_id, fase_ela=self.fase_ela,
                             duration=self.duracion, sampling_rate=self.sampling_rate["spo2"]).simular()
        df_resp = SensorRespiracion(self.paciente_id, fase_ela=self.fase_ela,
                                    duration=self.duracion, sampling_rate=self.sampling_rate["resp"]).simular()
        df_imu = SensorIMU(self.paciente_id, fase_ela=self.fase_ela,
                           duration=self.duracion, sampling_rate=self.sampling_rate["imu"], activity_period=None,
                           inmovilidad_period=None).simular()
        df_sueno = SensorSueno(self.paciente_id, fase_ela=self.fase_ela,
                               duration=self.duracion, sampling_rate=self.sampling_rate["sueno"]).simular()

        # Etiquetas
        etiquetador = EventLabeler(df_spo2, df_resp, df_imu, df_sueno)
        df_etiquetas = etiquetador.label_all()

        # Ordenar
        df_spo2 = df_spo2.sort_values("timestamp")
        df_resp = df_resp.sort_values("timestamp")
        df_imu = df_imu.sort_values("timestamp")
        df_sueno = df_sueno.sort_values("timestamp")
        df_etiquetas = df_etiquetas.sort_values("timestamp")

        # Merges con sufijos únicos para evitar columnas duplicadas idénticas
        df_merge = pd.merge_asof(df_imu, df_spo2, on="timestamp", direction="nearest",
                                 tolerance=pd.Timedelta("500ms"), suffixes=('_imu', '_spo2'))
        df_merge = pd.merge_asof(df_merge, df_resp, on="timestamp", direction="nearest",
                                 tolerance=pd.Timedelta("500ms"), suffixes=('', '_resp'))
        df_merge = pd.merge_asof(df_merge, df_sueno, on="timestamp", direction="nearest",
                                 tolerance=pd.Timedelta("500ms"), suffixes=('', '_sueno'))
        df_merge = pd.merge_asof(df_merge, df_etiquetas, on="timestamp", direction="nearest",
                                 tolerance=pd.Timedelta("500ms"), suffixes=('', '_etiqueta'))

        # Renombrar paciente_id_imu y fase_ela_imu a base
        df_merge.rename(columns={'paciente_id_imu': 'paciente_id', 'fase_ela_imu': 'fase_ela'}, inplace=True)

        # Limpiar columnas repetidas (mismo nombre exacto)
        df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]

        # Eliminar columnas paciente_id y fase_ela de sensores con sufijos
        cols_to_drop = [col for col in df_merge.columns if col not in ['paciente_id', 'fase_ela', 'timestamp'] and
                        (col.startswith('paciente_id') or col.startswith('fase_ela'))]
        df_merge.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # Reordenar columnas para claridad
        cols = df_merge.columns.tolist()
        for c in ['timestamp', 'paciente_id', 'fase_ela']:
            cols.remove(c)
        cols = ['timestamp', 'paciente_id', 'fase_ela'] + cols
        df_merge = df_merge[cols]

        return df_merge
