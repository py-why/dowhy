from dowhy.gcm import config
from dowhy.gcm.config import disable_progress_bars, enable_progress_bars, set_default_n_jobs


def test_when_enable_and_disable_progress_bars_then_global_parameter_set_correctly():
    disable_progress_bars()
    assert not config.show_progress_bars

    enable_progress_bars()
    assert config.show_progress_bars


def test_whe_set_default_n_jobs_then_global_parameter_set_correctly():
    tmp_default = config.default_n_jobs
    set_default_n_jobs(42)

    assert config.default_n_jobs == 42
    set_default_n_jobs(tmp_default)  # Restore default
