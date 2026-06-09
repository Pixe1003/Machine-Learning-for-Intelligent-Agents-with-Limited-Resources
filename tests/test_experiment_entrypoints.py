import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    REPO_ROOT / "experiments" / "phase1_exploratory.py",
    REPO_ROOT / "experiments" / "phase1_extended.py",
    REPO_ROOT / "experiments" / "phase2_core.py",
    REPO_ROOT / "experiments" / "phase3_stretch.py",
    REPO_ROOT / "experiments" / "reproduce_all.py",
]


def _direct_script_probe(script_path: Path) -> subprocess.CompletedProcess[str]:
    probe = f"""
import importlib.util
import sys
from pathlib import Path

repo_root = Path({str(REPO_ROOT)!r})
script_path = Path({str(script_path)!r})
sys.path = [str(script_path.parent)] + [
    entry for entry in sys.path if Path(entry or ".").resolve() != repo_root.resolve()
]
spec = importlib.util.spec_from_file_location("entrypoint_probe", script_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
"""
    return subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


@pytest.mark.parametrize("script_path", ENTRYPOINTS, ids=lambda path: path.stem)
def test_experiment_entrypoints_support_direct_script_loading(script_path: Path) -> None:
    result = _direct_script_probe(script_path)
    assert result.returncode == 0, result.stderr
