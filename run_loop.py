import subprocess

while True:

    result = subprocess.run(
        ["python", "indicators_REAL_FAST.py"]
    )

    if result.returncode != 0:
        break