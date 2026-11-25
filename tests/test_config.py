

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


def test_experiment_configs_load():
    """every experiment file (and the documented combination) loads and validates against the current default"""
    from climate_scenarios_2050.core import config as C

    files = sorted((C.CONFIG_DIR / "experiments").glob("*.yaml"))
    combos = [[f] for f in files] + [[C.CONFIG_DIR / "experiments" / n for n in ["south_america_pdo.yaml", "pdo_from_tas.yaml"]]]
    for combo in combos:
        cfg = C.load_config(combo)
        C.validate_indices(cfg)
        for name in C.index_names(cfg):
            C.index_cfg(cfg, name)


def test_read_only_prepare_writes_nothing(tmp_path):
    from climate_scenarios_2050.core.io import StepOutput, read_only

    cfg = {"paths": {"outputs": {"root": str(tmp_path)}}, "linear_model": {"min_samples": 20}}
    with read_only():
        out = StepOutput(cfg, "linear_model", sections=["linear_model"]).prepare()
    assert not out.dir.exists()
    assert StepOutput(cfg, "linear_model", sections=["linear_model"]).prepare().hash == out.hash
    assert (out.dir / "config.yaml").exists()
