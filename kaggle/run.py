from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import os
import shutil
import subprocess
import sys
import time
import tomllib

import torch
from packaging.requirements import Requirement


# =============================================================================
# INITIAL ENVIRONMENT
# =============================================================================

print(
    "GPU:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU",
    flush=True,
)
print(
    "GPU count:",
    torch.cuda.device_count(),
    flush=True,
)
print(
    "PyTorch:",
    torch.__version__,
    flush=True,
)
print(
    "CUDA:",
    torch.version.cuda,
    flush=True,
)
print(
    "cuDNN:",
    torch.backends.cudnn.version(),
    flush=True,
)


# =============================================================================
# PATHS
# =============================================================================

REPO_URL = "https://github.com/inori6/rxrx1.git"

WORK_DIR = Path("/kaggle/working")
PROJECT_DIR = WORK_DIR / "rxrx1"


# =============================================================================
# REVISE
# =============================================================================

GIT_REF = "master"

CONFIGS = [
    "configs/model_baseline.yaml",
]

HPO_STUDY_NAME = "model_baseline_hpo"
HPO_TIMEOUT_HOURS = 8
HPO_MAX_TRIALS = 100

# =============================================================================


# =============================================================================
# W&B SECRET
# =============================================================================

WANDB_KEY_CANDIDATES = [
    Path(
        "/kaggle/input/datasets/maributa/"
        "rxrx1-wandb-secret/"
        "wandb_api_key.txt"
    ),
    Path(
        "/kaggle/input/"
        "rxrx1-wandb-secret/"
        "wandb_api_key.txt"
    ),
]

WANDB_KEY_PATHS = [
    path
    for path in WANDB_KEY_CANDIDATES
    if path.is_file()
]

if len(WANDB_KEY_PATHS) == 0:
    raise FileNotFoundError(
        "W&B API key not found.\n"
        f"Checked: {WANDB_KEY_CANDIDATES}"
    )

if len(WANDB_KEY_PATHS) > 1:
    raise RuntimeError(
        "Multiple W&B API keys found.\n"
        f"Found: {WANDB_KEY_PATHS}"
    )

WANDB_KEY_PATH = WANDB_KEY_PATHS[0]

print(
    "W&B key path:",
    WANDB_KEY_PATH,
    flush=True,
)


# =============================================================================
# COMMAND
# =============================================================================

def run_command(command, cwd=None, check=True):
    print(flush=True)
    print(
        "=" * 80,
        flush=True,
    )
    print(
        "Running:",
        flush=True,
    )
    print(
        " ".join(map(str, command)),
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
    )


# =============================================================================
# SECRET
# =============================================================================

def setup_secrets():
    if not WANDB_KEY_PATH.exists():
        raise FileNotFoundError(
            f"W&B API key file not found: "
            f"{WANDB_KEY_PATH}"
        )

    key = WANDB_KEY_PATH.read_text(
        encoding="utf-8"
    ).strip()

    if not key:
        raise RuntimeError(
            "W&B API key file is empty."
        )

    os.environ["WANDB_API_KEY"] = key

    print(
        "WANDB_API_KEY loaded: True",
        flush=True,
    )


# =============================================================================
# GIT
# =============================================================================

def clone_repo():
    if PROJECT_DIR.exists():
        print(
            f"Removing existing project directory: "
            f"{PROJECT_DIR}",
            flush=True,
        )

        shutil.rmtree(
            PROJECT_DIR
        )

    run_command(
        [
            "git",
            "clone",
            REPO_URL,
            str(PROJECT_DIR),
        ],
        cwd=WORK_DIR,
    )


def checkout_git_ref():
    print(
        f"Checking out Git ref: "
        f"{GIT_REF}",
        flush=True,
    )

    run_command(
        [
            "git",
            "checkout",
            GIT_REF,
        ],
        cwd=PROJECT_DIR,
    )

    run_command(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=PROJECT_DIR,
    )


# =============================================================================
# ENVIRONMENT DIAGNOSTIC
# =============================================================================

def diagnose_environment(label):
    print(flush=True)
    print(
        "=" * 80,
        flush=True,
    )
    print(
        label,
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    run_command(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print('Python:', __import__('sys').executable); "
                "print('PyTorch:', torch.__version__); "
                "print('CUDA version:', torch.version.cuda); "
                "print('CUDA available:', torch.cuda.is_available()); "
                "print('GPU count:', torch.cuda.device_count()); "
                "print("
                "'GPU 0:', "
                "torch.cuda.get_device_name(0) "
                "if torch.cuda.is_available() "
                "else 'N/A'"
                ")"
            ),
        ],
        cwd=PROJECT_DIR,
    )


# =============================================================================
# PROJECT ENVIRONMENT
# =============================================================================

def install_project():
    pyproject_path = (
        PROJECT_DIR
        / "pyproject.toml"
    )

    if not pyproject_path.is_file():
        raise FileNotFoundError(
            f"pyproject.toml not found: "
            f"{pyproject_path}"
        )

    with pyproject_path.open(
        "rb"
    ) as file:
        pyproject = tomllib.load(
            file
        )

    requirements = (
        pyproject
        .get("project", {})
        .get("dependencies", [])
    )

    missing = []

    print(flush=True)
    print(
        "=" * 80,
        flush=True,
    )
    print(
        "Checking Kaggle environment...",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    for requirement_text in requirements:
        requirement = Requirement(
            requirement_text
        )

        try:
            installed_version = version(
                requirement.name
            )

            print(
                f"FOUND   | "
                f"{requirement.name}"
                f"=={installed_version}",
                flush=True,
            )

            if (
                requirement.specifier
                and installed_version
                not in requirement.specifier
            ):
                print(
                    f"WARNING | "
                    f"Kaggle has "
                    f"{requirement.name}"
                    f"=={installed_version}, "
                    f"but project requests "
                    f"{requirement_text}. "
                    f"Keeping Kaggle version.",
                    flush=True,
                )

        except PackageNotFoundError:
            print(
                f"MISSING | "
                f"{requirement_text}",
                flush=True,
            )

            missing.append(
                requirement_text
            )

    if missing:
        print(flush=True)
        print(
            f"Installing "
            f"{len(missing)} "
            f"missing package(s)...",
            flush=True,
        )

        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *missing,
            ],
            cwd=PROJECT_DIR,
        )

    else:
        print(flush=True)
        print(
            "All project dependencies "
            "are already available.",
            flush=True,
        )

    src_path = (
        PROJECT_DIR
        / "src"
    )

    current_pythonpath = (
        os.environ.get(
            "PYTHONPATH",
            "",
        )
    )

    if current_pythonpath:
        os.environ["PYTHONPATH"] = (
            f"{src_path}"
            f"{os.pathsep}"
            f"{current_pythonpath}"
        )
    else:
        os.environ["PYTHONPATH"] = (
            str(src_path)
        )

    print(flush=True)
    print(
        "Project source added "
        "to PYTHONPATH:",
        flush=True,
    )
    print(
        os.environ["PYTHONPATH"],
        flush=True,
    )


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment(config_path):
    print(flush=True)
    print(
        "#" * 80,
        flush=True,
    )
    print(
        f"STARTING: "
        f"{config_path}",
        flush=True,
    )
    print(
        "#" * 80,
        flush=True,
    )

    config_file = (
        PROJECT_DIR
        / config_path
    )

    if not config_file.exists():
        print(
            f"CONFIG NOT FOUND: "
            f"{config_file}",
            flush=True,
        )

        return False, None

    start_time = time.time()

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/hpo.py",
            "--config",
            config_path,
            "--study-name",
            HPO_STUDY_NAME,
            "--timeout-hours",
            str(HPO_TIMEOUT_HOURS),
            "--max-trials",
            str(HPO_MAX_TRIALS),
        ],
        cwd=PROJECT_DIR,
    )

    runtime_minutes = (
        time.time()
        - start_time
    ) / 60

    if result.returncode == 0:
        print(
            f"SUCCESS: "
            f"{config_path}",
            flush=True,
        )

        success = True

    else:
        print(
            f"FAILED: "
            f"{config_path}",
            flush=True,
        )

        print(
            f"Return code: "
            f"{result.returncode}",
            flush=True,
        )

        success = False

    print(
        f"Runtime: "
        f"{runtime_minutes:.2f} min",
        flush=True,
    )

    return (
        success,
        runtime_minutes,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(
        "=" * 80,
        flush=True,
    )
    print(
        "RxRx1 Kaggle Experiment Runner",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    setup_secrets()

    clone_repo()

    checkout_git_ref()

    diagnose_environment(
        "ENVIRONMENT BEFORE PROJECT SETUP"
    )

    install_project()

    diagnose_environment(
        "ENVIRONMENT AFTER PROJECT SETUP"
    )

    print(
        f"Repository : "
        f"{REPO_URL}",
        flush=True,
    )

    print(
        f"Git ref    : "
        f"{GIT_REF}",
        flush=True,
    )

    print(
        f"Experiments: "
        f"{len(CONFIGS)}",
        flush=True,
    )

    results = []

    total_start = time.time()

    for i, config_path in enumerate(
        CONFIGS,
        start=1,
    ):
        print(flush=True)

        print(
            "=" * 80,
            flush=True,
        )

        print(
            f"EXPERIMENT "
            f"{i}/{len(CONFIGS)}",
            flush=True,
        )

        print(
            "=" * 80,
            flush=True,
        )

        try:
            success, runtime = (
                run_experiment(
                    config_path
                )
            )

        except Exception as exc:
            print(
                f"UNEXPECTED ERROR: "
                f"{exc}",
                flush=True,
            )

            print(
                "Continuing to "
                "next experiment...",
                flush=True,
            )

            success = False
            runtime = None

        results.append(
            {
                "config": config_path,
                "success": success,
                "runtime": runtime,
            }
        )

    total_runtime = (
        time.time()
        - total_start
    ) / 60

    print(flush=True)

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "SUMMARY",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    for result in results:
        status = (
            "SUCCESS"
            if result["success"]
            else "FAILED"
        )

        runtime = (
            "N/A"
            if result["runtime"]
            is None
            else (
                f"{result['runtime']:.2f} min"
            )
        )

        print(
            f"{status:8} | "
            f"{runtime:12} | "
            f"{result['config']}",
            flush=True,
        )

    print(flush=True)

    print(
        f"Total runtime: "
        f"{total_runtime:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()