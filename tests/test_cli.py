import subprocess
import sys

from dls_fast_feedback import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "dls_fast_feedback", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
