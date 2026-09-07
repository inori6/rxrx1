from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import tomllib

import torch
from packaging.requirements import Requirement


# Change only this section between runs.
REPO_URL = "https://github.com/inori6/rxrx1.git"
GIT_REF = "master"
RUN_MODE = "train"

CONFIGS = [
    'configs/film_position/celltype_film_mid.yaml',

]

HPO_SCRIPT = "scripts/hpo_fusion.py"
HPO_STUDY_NAME = "well_film_lr_hpo"
HPO_TIMEOUT_HOURS = 5.5
HPO_MAX_TRIALS = 10

WORK_DIR = Path("/kaggle/working")
PROJECT_DIR = WORK_DIR / "rxrx1"
WANDB_KEY_CANDIDATES = [
    Path("/kaggle/input/datasets/maributa/rxrx1-wandb-secret/wandb_api_key.txt"),
    Path("/kaggle/input/rxrx1-wandb-secret/wandb_api_key.txt"),
]


def section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}", flush=True)


def run(command, *, cwd=PROJECT_DIR, check=True):
    print("Running:", " ".join(map(str, command)), flush=True)
    return subprocess.run(command, cwd=cwd, check=check)


def setup_wandb_key():
    paths = [path for path in WANDB_KEY_CANDIDATES if path.is_file()]
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one W&B key; found {len(paths)}. "
            f"Checked: {WANDB_KEY_CANDIDATES}"
        )
    key = paths[0].read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"W&B API key file is empty: {paths[0]}")
    os.environ["WANDB_API_KEY"] = key
    print(f"W&B key loaded from: {paths[0]}", flush=True)


def prepare_repository():
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    run(["git", "clone", REPO_URL, str(PROJECT_DIR)], cwd=WORK_DIR)
    run(["git", "checkout", GIT_REF])
    run(["git", "rev-parse", "HEAD"])


def install_missing_dependencies():
    pyproject_path = PROJECT_DIR / "pyproject.toml"
    if not pyproject_path.is_file():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject_path}")

    with pyproject_path.open("rb") as file:
        requirements = tomllib.load(file).get("project", {}).get("dependencies", [])

    missing = []
    for text in requirements:
        requirement = Requirement(text)
        try:
            installed = version(requirement.name)
            print(f"FOUND   | {requirement.name}=={installed}", flush=True)
            if requirement.specifier and installed not in requirement.specifier:
                print(f"WARNING | keeping {installed}; project requests {text}", flush=True)
        except PackageNotFoundError:
            print(f"MISSING | {text}", flush=True)
            missing.append(text)

    if missing:
        run([sys.executable, "-m", "pip", "install", *missing])

    src_path = str(PROJECT_DIR / "src")
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{previous}" if previous else src_path
    )


def print_environment():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Python : {sys.executable}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA   : {torch.version.cuda}", flush=True)
    print(f"cuDNN  : {torch.backends.cudnn.version()}", flush=True)
    print(f"GPU    : {gpu} ({torch.cuda.device_count()} device(s))", flush=True)


def build_experiment_command(config_path):
    common = ["--config", config_path]
    if RUN_MODE == "train":
        return [sys.executable, "-u", "scripts/train.py", *common]
    if RUN_MODE == "hpo":
        return [
            sys.executable,
            "-u",
            HPO_SCRIPT,
            *common,
            "--study-name",
            HPO_STUDY_NAME,
            "--timeout-hours",
            str(HPO_TIMEOUT_HOURS),
            "--max-trials",
            str(HPO_MAX_TRIALS),
        ]
    raise ValueError(f"RUN_MODE must be 'train' or 'hpo', got {RUN_MODE!r}")


def run_experiment(config_path):
    config_file = PROJECT_DIR / config_path
    if not config_file.is_file():
        print(f"CONFIG NOT FOUND: {config_file}", flush=True)
        return False, None

    section(f"STARTING: {config_path}")
    started = time.time()
    result = run(build_experiment_command(config_path), check=False)
    minutes = (time.time() - started) / 60
    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"{status}: {config_path} ({minutes:.2f} min)", flush=True)
    return result.returncode == 0, minutes


def main():
    section("RxRx1 Kaggle Experiment Runner")
    setup_wandb_key()
    prepare_repository()
    section("Environment")
    print_environment()
    install_missing_dependencies()

    print(f"Repository : {REPO_URL}", flush=True)
    print(f"Git ref    : {GIT_REF}", flush=True)
    print(f"Run mode   : {RUN_MODE}", flush=True)
    print(f"Experiments: {len(CONFIGS)}", flush=True)

    results = []
    started = time.time()
    for index, config_path in enumerate(CONFIGS, 1):
        section(f"EXPERIMENT {index}/{len(CONFIGS)}")
        try:
            success, minutes = run_experiment(config_path)
        except Exception as exc:
            print(f"UNEXPECTED ERROR: {exc}", flush=True)
            success, minutes = False, None
        results.append((config_path, success, minutes))

    section("SUMMARY")
    for config_path, success, minutes in results:
        status = "SUCCESS" if success else "FAILED"
        runtime = "N/A" if minutes is None else f"{minutes:.2f} min"
        print(f"{status:8} | {runtime:12} | {config_path}", flush=True)
    print(f"Total runtime: {(time.time() - started) / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()