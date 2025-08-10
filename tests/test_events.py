import pandas as pd
import pytest
from etiquetas.EventLabeler import EventLabeler


@pytest.fixture
# Datos de prueba para eventos
def test_events_data():
    # Timestamps cada segundo
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=10, freq="1s")

    # Datos para SpO2: primero normal, luego forzamos hipoxia
    df_spo2 = pd.DataFrame({
        "timestamp": timestamps,
        "spo2": [95, 94, 89, 88, 87, 95, 94, 91, 89, 88],
        "frecuencia_cardiaca": [80, 85, 90, 102, 105, 85, 82, 99, 101, 110]
    })

    # Datos para respiración: eventos de hipoventilación
    df_resp = pd.DataFrame({
        "timestamp": timestamps,
        "evento_hipoventilacion": [0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    })

    # Datos para IMU: eventos de inmovilidad
    df_imu = pd.DataFrame({
        "timestamp": timestamps,
        "evento_inmovilidad": [1]*9 + [0]  # 9 seg inmóviles → supera 50 seg en ventana de 60
    })

    # Datos para sueño: fragmentación forzada
    df_sueno = pd.DataFrame({
        "timestamp": timestamps,
        "fase_sueno": ["AWAKE"]*6 + ["NREM"]*4,
        "evento_fragmentacion": [0, 1, 1, 1, 1, 2, 0, 0, 0, 0]
    })

    return df_spo2, df_resp, df_imu, df_sueno


# Test para verificar la correcta etiquetación de eventos
def test_label_all(test_events_data):
    df_spo2, df_resp, df_imu, df_sueno = test_events_data
    labeler = EventLabeler(df_spo2, df_resp, df_imu, df_sueno)
    result = labeler.label_all()

    # Comprobar que las columnas de eventos existen
    expected_cols = [
        "timestamp",
        "inmovilidad_sostenida",
        "hipoxia_sostenida",
        "hipovent_sostenida",
        "frag_sueno_sostenido",
        "empeoramiento"
    ]
    for col in expected_cols:
        assert col in result.columns, f"Falta columna {col}"

    # Validar que al menos un valor de cada evento sea 1
    assert result["hipoxia_sostenida"].sum() > 0
    assert result["hipovent_sostenida"].sum() > 0
    assert result["frag_sueno_sostenido"].sum() > 0

    # Validar que empeoramiento es la combinación de los eventos
    assert result["empeoramiento"].sum() > 0, "No se detectó empeoramiento cuando debería"
