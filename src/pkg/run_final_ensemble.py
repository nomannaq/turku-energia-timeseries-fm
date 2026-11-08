"""Reproduce the supplied final ensemble with the project's two existing environments."""
from concurrent.futures import ThreadPoolExecutor
import subprocess

from ensemble_spec import ROOT, OUT, WORK


def run(script, env=".venv", *args):
    command = [str(ROOT / env / "bin/python"), "-u", str(ROOT / "src/pkg" / script), *args]
    print("Running", script, *args, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    run("ensemble_leak_test.py", ".venv", "features")
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run, "ens_members.py"), pool.submit(run, "fm_run.py", ".venv-chronos")]
        for job in jobs:
            job.result()
    run("ensemble_leak_test.py", ".venv", "lgb")
    run("ensemble_leak_test.py", ".venv-chronos", "chronos")
    run("ensemble_fold_scores.py")
    run("fm_blend3.py")
    run("verify_ensemble.py")
