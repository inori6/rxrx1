from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

import torch


print(
    "GPU:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    flush=True,
)
print("GPU count:", torch.cuda.device_count(), flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("CUDA:", torch.version.cuda, flush=True)
print("cuDNN:", torch.backends.cudnn.version(), flush=True)


REPO_URL = "https://github.com/inori6/rxrx1.git"
WORK_DIR = Path("/kaggle/working")
PROJECT_DIR = WORK_DIR / "rxrx1"


# =============================================================================
# REVISE
# =============================================================================

GIT_REF = "experiment/normalization"

CONFIGS = [
    "configs/normalization/efficientnet_b2_norm/grouping/"
    "effb2_norm_src-sample_pop-np_sp-global_ch-shared_pol-trainonly_pos-pre.yaml",
]

# =============================================================================


WANDB_KEY_CANDIDATES = [
    Path(
        "/kaggle/input/datasets/maributa/"
        "rxrx1-wandb-secret/wandb_api_key.txt"
    ),
    Path(
        "/kaggle/input/"
        "rxrx1-wandb-secret/wandb_api_key.txt"
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


def run_command(command, cwd=None, check=True):
    print(flush=True)
    print("=" * 80, flush=True)
    print("Running:", flush=True)
    print(" ".join(map(str, command)), flush=True)
    print("=" * 80, flush=True)

    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
    )


def setup_secrets():
    if not WANDB_KEY_PATH.exists():
        raise FileNotFoundError(
            f"W&B API key file not found: {WANDB_KEY_PATH}"
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


def clone_repo():
    if PROJECT_DIR.exists():
        print(
            f"Removing existing project directory: {PROJECT_DIR}",
            flush=True,
        )
        shutil.rmtree(PROJECT_DIR)

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
        f"Checking out Git ref: {GIT_REF}",
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


def install_project():
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
        ],
        cwd=PROJECT_DIR,
    )


def run_experiment(config_path):
    print(flush=True)
    print("#" * 80, flush=True)
    print(
        f"STARTING: {config_path}",
        flush=True,
    )
    print("#" * 80, flush=True)

    config_file = PROJECT_DIR / config_path

    if not config_file.exists():
        print(
            f"CONFIG NOT FOUND: {config_file}",
            flush=True,
        )
        return False, None

    start_time = time.time()

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/train.py",
            "--config",
            config_path,
        ],
        cwd=PROJECT_DIR,
    )

    runtime_minutes = (
        time.time() - start_time
    ) / 60

    if result.returncode == 0:
        print(
            f"SUCCESS: {config_path}",
            flush=True,
        )
        success = True
    else:
        print(
            f"FAILED: {config_path}",
            flush=True,
        )
        print(
            f"Return code: {result.returncode}",
            flush=True,
        )
        success = False

    print(
        f"Runtime: {runtime_minutes:.2f} min",
        flush=True,
    )

    return success, runtime_minutes


def main():
    print("=" * 80, flush=True)
    print(
        "RxRx1 Kaggle Experiment Runner",
        flush=True,
    )
    print("=" * 80, flush=True)

    setup_secrets()
    clone_repo()
    checkout_git_ref()
    install_project()

    print(
        f"Repository : {REPO_URL}",
        flush=True,
    )
    print(
        f"Git ref    : {GIT_REF}",
        flush=True,
    )
    print(
        f"Experiments: {len(CONFIGS)}",
        flush=True,
    )

    results = []
    total_start = time.time()

    for i, config_path in enumerate(
        CONFIGS,
        start=1,
    ):
        print(flush=True)
        print("=" * 80, flush=True)
        print(
            f"EXPERIMENT {i}/{len(CONFIGS)}",
            flush=True,
        )
        print("=" * 80, flush=True)

        try:
            success, runtime = run_experiment(
                config_path
            )
        except Exception as exc:
            print(
                f"UNEXPECTED ERROR: {exc}",
                flush=True,
            )
            print(
                "Continuing to next experiment...",
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
        time.time() - total_start
    ) / 60

    print(flush=True)
    print("=" * 80, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 80, flush=True)

    for result in results:
        status = (
            "SUCCESS"
            if result["success"]
            else "FAILED"
        )

        runtime = (
            "N/A"
            if result["runtime"] is None
            else f"{result['runtime']:.2f} min"
        )

        print(
            f"{status:8} | "
            f"{runtime:12} | "
            f"{result['config']}",
            flush=True,
        )

    print(flush=True)
    print(
        f"Total runtime: {total_runtime:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()