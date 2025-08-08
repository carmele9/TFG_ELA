import pytest
import pandas as pd
from sensores.SensorIMU import SensorIMU
from sensores.SensorRespiracion import SensorRespiracion
from sensores.SensorSpO2 import SensorSpO2
from sensores.SensorSueno import SensorSueno


@pytest.mark.parametrize("SensorClass, expected_cols, value_checks", [
    (SensorIMU, ["timestamp", "paciente_id", "acc_x", "acc_y", "acc_z", "giro_x", "giro_y", "giro_z"], None),
    (SensorRespiracion, ["timestamp", "paciente_id", "frecuencia_respiratoria"], lambda df: df["frecuencia_respiratoria"].between(8, 40).all()),
    (SensorSpO2, ["timestamp", "paciente_id", "SpO2"], lambda df: df["SpO2"].between(70, 100).all()),
    (SensorSueno, ["timestamp", "paciente_id", "fase_sueno"], None)
])
def test_sensor_output(SensorClass, expected_cols, value_checks):
    sensor = SensorClass(paciente_id="PAC_001")
    df = sensor.simular()

    # Formato esperado
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == expected_cols
    assert df.isna().sum().sum() == 0
    assert len(df) == 60  # 1 fila por segundo

    # Valores realistas si aplica
    if value_checks:
        assert value_checks(df)


def test_sensor_sueno():
    sensor = SensorSueno(paciente_id="PAC_001")
    df = sensor.simular()
    assert df["fase_sueno"].nunique() == 5  # 5 fases diferentes
    assert df["fase_sueno"].between(0, 4).all()  # Fases entre 0 y 4
    assert len(df) == 60  # 1 fila por segundo


def test_sensor_imu():
    sensor = SensorIMU(paciente_id="PAC_001")
    df = sensor.simular()
    assert all(col in df.columns for col in ["acc_x", "acc_y", "acc_z", "giro_x", "giro_y", "giro_z"])
    assert df["acc_x"].between(-10, 10).all()
    assert df["acc_y"].between(-10, 10).all()
    assert df["acc_z"].between(-10, 10).all()
    assert df["giro_x"].between(-180, 180).all()
    assert df["giro_y"].between(-180, 180).all()
    assert df["giro_z"].between(-180, 180).all()


def test_sensor_respiracion():
    sensor = SensorRespiracion(paciente_id="PAC_001")
    df = sensor.simular()
    assert "frecuencia_respiratoria" in df.columns
    assert df["frecuencia_respiratoria"].between(8, 40).all()  # Valores realistas
    assert len(df) == 60  # 1 fila por segundo


def test_sensor_spo2():
    sensor = SensorSpO2(paciente_id="PAC_001")
    df = sensor.simular()
    eventos = df[df["evento"] == "hipoxia"]
    assert all(df["SpO2"] < 90 for _, df in eventos.iterrows())
