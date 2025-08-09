import pytest
import pandas as pd
from sensores.SensorIMU import SensorIMU
from sensores.SensorRespiracion import SensorRespiracion
from sensores.SensorSpO2 import SensorSpO2
from sensores.SensorSueno import SensorSueno


# Test inicialización de sensores
@pytest.mark.parametrize("SensorClass", [SensorIMU, SensorRespiracion, SensorSpO2, SensorSueno])
def test_sensor_initialization(SensorClass):
    sensor = SensorClass(paciente_id="PAC_001")
    assert sensor.paciente_id == "PAC_001"
    df = sensor.simular()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


# Test de formato y valores básicos
@pytest.mark.parametrize("SensorClass, expected_cols, value_checks", [
    (SensorIMU, ['timestamp', 'paciente_id', 'fase_ela', 'aceleracion_x', 'aceleracion_y', 'aceleracion_z',
                 'magnitud_movimiento', 'actividad_estimada', 'evento_inmovilidad', 'estado'], None),
    (SensorRespiracion, ['timestamp', 'paciente_id', 'fase_ela', 'senal_respiratoria', 'frecuencia_respiratoria',
                         'variabilidad_respiratoria', 'amplitud_instante', 'evento_hipoventilacion', 'evento_brp'],
     lambda df: df["frecuencia_respiratoria"].between(8, 40).all()),
    (SensorSpO2, ['timestamp', 'paciente_id','fase_ela', 'spo2', 'frecuencia_cardiaca','pulsatility_index',
                  'evento_hipoxia'],
     lambda df: df["spo2"].between(70, 100).all()),
    (SensorSueno, ['timestamp', 'paciente_id', 'fase_ela', 'senal_sueno','fase_sueno', 'evento_fragmentacion'],
     lambda df: df["fase_sueno"].nunique() == 4 and df["evento_fragmentacion"].isin([0, 1]).all())
])
def test_sensor_format_and_values(SensorClass, expected_cols, value_checks):
    sensor = SensorClass(paciente_id="PAC_001")
    df = sensor.simular()
    assert all(col in df.columns for col in expected_cols)
    assert df["timestamp"].dtype == "datetime64[ns]"
    assert df["paciente_id"].dtype == "object"
    assert df["fase_ela"].dtype == "int64"
    if value_checks:
        assert value_checks(df)


# Test específico sueño
def test_sensor_sueno():
    sensor = SensorSueno(paciente_id="PAC_001")
    df = sensor.simular()
    assert df["fase_sueno"].nunique() == 4
    assert df["evento_fragmentacion"].isin([0, 1]).all()
    assert df["timestamp"].dtype == "datetime64[ns]"
    assert df["paciente_id"].dtype == "object"
    assert df["fase_ela"].dtype == "int64"
    assert df["senal_sueno"].dtype == "float64"
    assert df["fase_sueno"].dtype == "object"
    assert df.isna().sum().sum() == 0


# Test específico IMU
def test_sensor_imu():
    sensor = SensorIMU(paciente_id="PAC_001")
    df = sensor.simular()
    assert all(col in df.columns for col in ["aceleracion_x", "aceleracion_y", "aceleracion_z"])
    assert df["aceleracion_x"].between(-10, 10).all()
    assert df["aceleracion_y"].between(-10, 10).all()
    assert df["aceleracion_z"].between(-10, 10).all()
    assert df["magnitud_movimiento"].between(0, 10).all()
    assert df["actividad_estimada"].between(0, 100).all()
    assert df["evento_inmovilidad"].isin([0, 1]).all()
    assert df["timestamp"].dtype == "datetime64[ns]"
    assert df["paciente_id"].dtype == "object"
    assert df["fase_ela"].dtype == "int64"
    assert df["aceleracion_x"].dtype == "float64"
    assert df.isna().sum().sum() == 0


# Test específico respiración
def test_sensor_respiracion():
    sensor = SensorRespiracion(paciente_id="PAC_001")
    df = sensor.simular()
    assert "frecuencia_respiratoria" in df.columns
    assert df["frecuencia_respiratoria"].between(8, 40).all()
    assert df["evento_hipoventilacion"].isin([0, 1]).all()
    assert df["evento_brp"].isin([0, 1]).all()
    assert df["senal_respiratoria"].dtype == "float64"
    assert df["variabilidad_respiratoria"].dtype == "float64"
    assert df["amplitud_instante"].dtype == "float64"
    assert df["timestamp"].dtype == "datetime64[ns]"
    assert df["paciente_id"].dtype == "object"
    assert df["fase_ela"].dtype == "int64"
    assert df["frecuencia_respiratoria"].isna().sum() == 0


# Test específico SpO2 (ajustado a columnas reales)
def test_sensor_spo2():
    sensor = SensorSpO2(paciente_id="PAC_001")
    df = sensor.simular()
    assert "evento_hipoxia" in df.columns
    assert df["frecuencia_cardiaca"].between(40, 180).all()
    assert df["spo2"].dtype == "float64"
    assert df["spo2"].between(70, 100).all()
    assert df["timestamp"].dtype == "datetime64[ns]"
    assert df["paciente_id"].dtype == "object"
    assert df["fase_ela"].dtype == "int64"
    assert df["spo2"].isna().sum() == 0


# Test de simulación de sensores
def test_sensor_simulation():
    sensor_imu = SensorIMU(paciente_id="PAC_001")
    df_imu = sensor_imu.simular()
    assert isinstance(df_imu, pd.DataFrame)
    assert len(df_imu) > 0

    sensor_resp = SensorRespiracion(paciente_id="PAC_001")
    df_resp = sensor_resp.simular()
    assert isinstance(df_resp, pd.DataFrame)
    assert len(df_resp) > 0

    sensor_spo2 = SensorSpO2(paciente_id="PAC_001")
    df_spo2 = sensor_spo2.simular()
    assert isinstance(df_spo2, pd.DataFrame)
    assert len(df_spo2) > 0

    sensor_sueno = SensorSueno(paciente_id="PAC_001")
    df_sueno = sensor_sueno.simular()
    assert isinstance(df_sueno, pd.DataFrame)
    assert len(df_sueno) > 0

