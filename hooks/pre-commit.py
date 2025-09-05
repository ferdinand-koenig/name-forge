#!/usr/bin/env python
import subprocess
import sys
import os

# Force UTF-8 encoding (Windows)
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PIPENV_VERBOSITY"] = "-1"

WIZARD = r"""
                    ____ 
                  .* *.
               __/_*_*(_
              / _______ \
             _\_)/___\(_/_ 
            / _((\- -/))_ \
            \ \())(-)(()/ /
             ' \(((()))/ '
            / ' \)).))/ ' \
           / _ \ - | - /_  \
          (   ( .;''';. .'  )
          _\"__ /    )\ __"/_
            \/  \   ' /  \/
             .'  '...' ' )
              / /  |  \ \
             / .   .   . \
            /   .     .   \
           /   /   |   \   \
         .'   /    b    '.  '.
     _.-'    /     Bb     '-. '-._ 
 _.-'       |      BBb       '-. '-. 
(________mrf\____.dBBBb.________)____)
"""

def run(command_list):
    """Run a subprocess command list and return the exit code."""
    return subprocess.run(command_list).returncode

# -----------------------------
# Get staged Python / notebook files
# -----------------------------
staged_files = subprocess.check_output(
    "git diff --cached --name-only", shell=True
).decode("utf-8").splitlines()

# Filter only Python / notebooks and exclude the hook itself
staged_files = [
    f for f in staged_files
    if f.endswith((".py", ".ipynb")) and f != "hooks/pre-commit.py"
]

if not staged_files:
    print("No Python or notebook files staged. The Wizard nods...")
    sys.exit(0)

# -----------------------------
# Auto-fix staged files
# -----------------------------
print("Running Black...")
run(["pipenv", "run", "black"] + staged_files)

print("Running isort...")
run(["pipenv", "run", "isort"] + staged_files)

print("Running autoflake...")
run([
    "pipenv", "run", "autoflake",
    "--in-place", "--remove-unused-variables", "--remove-all-unused-imports", "-r"
] + staged_files)

# Stage any changes made by formatters
subprocess.run(["git", "add"] + staged_files)

# -----------------------------
# Final lint check
# -----------------------------
if run(["pipenv", "run", "flake8"] + staged_files) != 0:
    print("The Wizard frowns! Your code still has linting issues:")
    print(WIZARD)
    print("Run 'pipenv run flake8' to fix manually.")
    sys.exit(1)

print("The Wizard approves! Your staged code is magical!")
print(WIZARD)
