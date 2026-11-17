import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("log_path", ["agent-stdio.log", "threat-detection/detection.log"])
def test_copilot_driver_path_without_runner_temp(monkeypatch, tmp_path, log_path):
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/repo-assist.lock.yml"
    command_line = next(
        line for line in workflow.read_text().splitlines() if "-- /bin/bash -c " in line and log_path in line
    )
    runner_temp = tmp_path / "runner temp"
    command_line = command_line.replace("${{ runner.temp }}", str(runner_temp))
    command = shlex.split(command_line)[3]
    monkeypatch.delenv("RUNNER_TEMP", raising=False)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'node() { printf "%s\\n" "$@"; }; cat() { printf "test prompt"; }; ' + command,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = result.stdout.splitlines()
    assert arguments[:2] == [str(runner_temp / "gh-aw/actions/copilot_driver.cjs"), "/usr/local/bin/copilot"]
    assert arguments[arguments.index("--add-dir", arguments.index("--add-dir") + 1) + 1] == str(tmp_path)
    assert arguments[-2:] == ["--prompt", "test prompt"]
