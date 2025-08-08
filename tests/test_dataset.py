from core.SimularDataset import SimuladorDataset


def test_combinacion_sensores(tmp_path):
    output_csv = tmp_path / "dataset.csv"
    sim = SimuladorDataset(paciente_id="PAC_001", fase_ela=1, duracion=600)
    df = sim.generar()

    # Guardado
    df.to_csv(output_csv, index=False)
    assert output_csv.exists()

    # Columnas y duplicados
    assert "timestamp" in df.columns
    assert "paciente_id" in df.columns
    assert df.duplicated(subset=["timestamp", "paciente_id"]).sum() == 0

    # Tamaño esperado
    assert df["paciente_id"].nunique() == 2

