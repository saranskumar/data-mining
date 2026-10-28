"""Tiny wrapper used by app/jobs.py: run a command and record its exit code in a file.

Usage: python _runner.py <exit_file> <command> [args ...]
The wrapper itself writes to the job log (stdout/stderr are inherited by the command).
"""
import subprocess
import sys

exit_file, *cmd = sys.argv[1:]
rc = subprocess.call(cmd)
with open(exit_file, "w", encoding="utf-8") as fh:
    fh.write(str(rc))
sys.exit(rc)
