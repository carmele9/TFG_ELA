import pandas as pd


class EventLabeler:
    def __init__(self, df_spo2, df_resp, df_imu, df_sueno):
        """
        Inicializa la clase con los cuatro DataFrames de los sensores simulados.
        Estos DataFrames deben contener variables fisiológicas generadas previamente.
        """
        self.spo2 = df_spo2
        self.resp = df_resp
        self.imu = df_imu
        self.sueno = df_sueno

    def label_spo2(self):
        """
        Detecta eventos de hipoxia:
        - SpO2 < 90% durante al menos 3 muestras consecutivas
        - O SpO2 < 92% junto con frecuencia cardíaca > 100 bpm
        Devuelve una columna binaria 'hipoxia_sostenida'.
        """
        s = self.spo2
        s['hipoxia_sostenida'] = (
                s['spo2'].rolling(window=3, min_periods=1).apply(lambda x: (x < 90).all()).fillna(0).astype(bool) |
                ((s['spo2'] < 92) & (s['frecuencia_cardiaca'] > 100))
        ).astype(int)
        return s['hipoxia_sostenida']

    def label_resp(self):
        """
        Detecta episodios de hipoventilación:
        - Se considera un evento si hay ≥ 5 episodios de hipoventilación en una ventana de 10 segundos.
        Devuelve una columna binaria 'hipovent_sostenido'.
        """
        r = self.resp
        r['hipovent_sostenido'] = (
            r['evento_hipoventilacion'].rolling(window=10, min_periods=1).sum() >= 5
        ).astype(int)
        return r['hipovent_sostenido']

    def label_imu(self):
        """
        Detecta inmovilidad prolongada:
        - Se considera un evento si hay ≥ 50 segundos inmóvil dentro de una ventana de 60 segundos.
        Devuelve una columna binaria 'inmovilidad_sostenida'.
        """
        m = self.imu
        m['inmovilidad_sostenida'] = (
            m['evento_inmovilidad'].rolling(window=60, min_periods=1).sum() >= 50
        ).astype(int)
        return m['inmovilidad_sostenida']

    def label_sueno(self):
        """
        Detecta sueño muy fragmentado:
        - Más del 50% del tiempo en los últimos 10 minutos (600 seg) en fase 'AWAKE'
        - O más de 5 cambios de fase en ese mismo intervalo
        Devuelve una columna binaria 'frag_sueno_sostenido'.
        """
        s = self.sueno
        awake = (s['fase_sueno'] == 'AWAKE').astype(int)
        frag = s['evento_fragmentacion'].rolling(window=600, min_periods=1).sum()
        s['frag_sueno_sostenido'] = ((awake.rolling(600).mean() > 0.5) | (frag > 5)).astype(int)
        return s['frag_sueno_sostenido']

    def label_all(self):
        """
        Aplica todas las funciones de etiquetado de eventos:
        - Combina hipoxia, hipoventilación, inmovilidad y fragmentación del sueño
        - Genera un DataFrame con una etiqueta global 'empeoramiento'
          que se activa si al menos uno de los eventos individuales está presente.
        Se usa merge_asof para alinear temporalmente los eventos más cercanos.
        """

        # Crear DataFrames individuales con timestamps y etiquetas
        df_spo2 = self.spo2[['timestamp']].copy()
        df_spo2['hipoxia_sostenida'] = self.label_spo2()

        df_resp = self.resp[['timestamp']].copy()
        df_resp['hipovent_sostenida'] = self.label_resp()

        df_imu = self.imu[['timestamp']].copy()
        df_imu['inmovilidad_sostenida'] = self.label_imu()

        df_sueno = self.sueno[['timestamp']].copy()
        df_sueno['frag_sueno_sostenido'] = self.label_sueno()

        # Ordenar por tiempo
        df_spo2 = df_spo2.sort_values("timestamp")
        df_resp = df_resp.sort_values("timestamp")
        df_imu = df_imu.sort_values("timestamp")
        df_sueno = df_sueno.sort_values("timestamp")

        # Usamos la señal de IMU como base por su mayor resolución
        df = df_imu.copy()

        # Alinear temporalmente con merge_asof (tolerancia de 1 segundo)
        df = pd.merge_asof(df, df_spo2, on="timestamp", direction="nearest", tolerance=pd.Timedelta("1s"))
        df = pd.merge_asof(df, df_resp, on="timestamp", direction="nearest", tolerance=pd.Timedelta("1s"))
        df = pd.merge_asof(df, df_sueno, on="timestamp", direction="nearest", tolerance=pd.Timedelta("1s"))

        # Rellenar posibles NaN con 0
        df.fillna(0, inplace=True)

        # Etiqueta global de empeoramiento
        df['empeoramiento'] = (
                df[['hipoxia_sostenida', 'hipovent_sostenida', 'inmovilidad_sostenida', 'frag_sueno_sostenido']].sum(
                    axis=1) >= 1
        ).astype(int)

        return df

