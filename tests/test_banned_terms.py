"""Run the banned-terms check as part of the test suite."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_banned_terms():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_banned_terms.py"), str(ROOT)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
