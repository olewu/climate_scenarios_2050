

def test_baseline_fit_window_follows_matching_window(tmp_path):
    from climate_scenarios_2050.core.config import load_config

    cfg = load_config()
    assert cfg["baselines"]["ar1"]["ar1_fair"]["fit_window"] == cfg["analogues"]["window"]
    assert cfg["baselines"]["ar1"]["ar1_all"]["fit_window"] == "all"
    override = tmp_path / "w.yaml"
    override.write_text("analogues:\n  window: 50\n")
    cfg = load_config([override])
    assert cfg["baselines"]["ar1"]["ar1_fair"]["fit_window"] == 50
    assert cfg["baselines"]["sine_trend"]["fit_window"] == 50
