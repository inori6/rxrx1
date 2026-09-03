from pathlib import Path
import shutil
import subprocess
import sys
import time
import os
from kaggle_secrets import UserSecretsClient


REPO_URL = "https://github.com/inori6/rxrx1.git"

# 可以写：
# "master"
# "experiment/augmentation"
# "v1.0"
# "a1b2c3d4..."


GIT_REF = "experiment/convergence"

WORK_DIR = Path("/kaggle/working")
PROJECT_DIR = WORK_DIR / "rxrx1"

CONFIGS = [
    "configs/convergence/efficientnet_b2_30e.yaml"
]


def run_command(command, cwd=None, check=True):
    print()
    print("=" * 80)
    print("Running:")
    print(" ".join(map(str, command)))
    print("=" * 80)

    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
    )


def clone_repo():
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
    print(f"Checking out Git ref: {GIT_REF}")

    run_command(
        [
            "git",
            "checkout",
            GIT_REF,
        ],
        cwd=PROJECT_DIR,
    )

    # 打印真正运行的 commit
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
    print()
    print("#" * 80)
    print(f"STARTING: {config_path}")
    print("#" * 80)

    config_file = PROJECT_DIR / config_path

    if not config_file.exists():
        print(f"CONFIG NOT FOUND: {config_file}")
        return False, None

    start_time = time.time()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            config_path,
        ],
        cwd=PROJECT_DIR,
    )

    runtime_minutes = (time.time() - start_time) / 60

    if result.returncode == 0:
        print(f"SUCCESS: {config_path}")
        success = True
    else:
        print(f"FAILED: {config_path}")
        print(f"Return code: {result.returncode}")
        success = False

    print(f"Runtime: {runtime_minutes:.2f} min")

    return success, runtime_minutes


def setup_secrets():
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    print("WANDB_API_KEY loaded:", bool(os.environ.get("WANDB_API_KEY")))


def main():
    print("=" * 80)
    print("RxRx1 Kaggle Experiment Runner")
    print("=" * 80)

    setup_secrets()
    clone_repo()
    checkout_git_ref()
    install_project()

    print(f"Repository : {REPO_URL}")
    print(f"Git ref    : {GIT_REF}")
    print(f"Experiments: {len(CONFIGS)}")

    results = []
    total_start = time.time()

    for i, config_path in enumerate(CONFIGS, start=1):
        print()
        print("=" * 80)
        print(f"EXPERIMENT {i}/{len(CONFIGS)}")
        print("=" * 80)

        try:
            success, runtime = run_experiment(config_path)

        except Exception as exc:
            print(f"UNEXPECTED ERROR: {exc}")
            print("Continuing to next experiment...")

            success = False
            runtime = None

        results.append(
            {
                "config": config_path,
                "success": success,
                "runtime": runtime,
            }
        )

    total_runtime = (time.time() - total_start) / 60

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"

        if result["runtime"] is None:
            runtime = "N/A"
        else:
            runtime = f"{result['runtime']:.2f} min"

        print(
            f"{status:8} | "
            f"{runtime:12} | "
            f"{result['config']}"
        )

    print()
    print(f"Total runtime: {total_runtime:.2f} min")


if __name__ == "__main__":
    main()
