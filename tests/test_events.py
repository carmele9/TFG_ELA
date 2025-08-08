from etiquetas.EventLabeler import EventLabeler
from sensores.SensorSpO2 import SensorSpO2
from sensores.SensorIMU import SensorIMU
from sensores.SensorSueno import SensorSueno
from sensores.SensorRespiracion import SensorRespiracion


def test_event_labeler():
    # Create instances of sensors
    sensor_spo2 = SensorSpO2(paciente_id="test_patient")
    sensor_imu = SensorIMU(paciente_id="test_patient")
    sensor_sueno = SensorSueno(paciente_id="test_patient")
    sensor_respiracion = SensorRespiracion(paciente_id="test_patient")

    # Create an instance of EventLabeler
    event_labeler = EventLabeler(sensor_spo2, sensor_imu, sensor_sueno, sensor_respiracion)

    # Test the label_event method
    event_labeler.label_all()

    assert event_labeler.spo2 is not None, "SpO2 data should not be None"
    assert event_labeler.imu is not None, "IMU data should not be None"
    assert event_labeler.sueno is not None, "Sleep data should not be None"
    assert event_labeler.resp is not None, "Respiration data should not be None"
    assert event_labeler.spo2.paciente_id == "test_patient", "SpO2 patient ID should match"
    assert event_labeler.imu.paciente_id == "test_patient", "IMU patientID should match"
    assert event_labeler.sueno.paciente_id == "test_patient", "Sleep patient ID should match"
    assert event_labeler.resp.paciente_id == "test_patient", "Respiration" "patient ID should match"
