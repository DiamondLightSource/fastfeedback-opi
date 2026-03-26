import subprocess
import sys

from fastfeedback_opi import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "fastfeedback_opi", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
