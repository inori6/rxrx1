from pathlib import Path

import wandb

def setup_wandb(config: dict, project_root: Path):
    wandb_config = config.get("wandb", {})

    if not wandb_config.get("enabled", False):
        return None

    run = wandb.init(
        project=wandb_config["project"],
        name=config["experiment"]["name"],
        config=config,
        mode=wandb_config["mode"],
        dir=str(project_root / wandb_config["dir"]),
    )

    return run

def update_wandb_git_info(
    run,
    git_commit,
    git_ref,
):
    if run is None:
        return

    run.config.update(
        {
            "git_commit": git_commit,
            "git_ref": git_ref,
        },
        allow_val_change=True,
    )


def log_wandb_epoch(
    run,
    epoch,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    runtime_minutes,
):
    if run is None:
        return

    run.log(
        {
            "epoch": epoch,

            "train/loss": train_loss,
            "train/acc": train_acc,

            "val/loss": val_loss,
            "val/acc": val_acc,

            "runtime/epoch_minutes": runtime_minutes,
        }
    )


def finish_wandb(
    run,
    results,
):
    if run is None:
        return

    run.summary.update(
        {
            "final_train_acc":
                results.final_train_acc,

            "final_train_loss":
                results.final_train_loss,

            "final_val_acc":
                results.final_val_acc,

            "final_val_loss":
                results.final_val_loss,

            "best_epoch":
                results.best_epoch,

            "best_train_acc":
                results.best_train_acc,

            "best_train_loss":
                results.best_train_loss,

            "best_val_acc":
                results.best_val_acc,

            "best_val_loss":
                results.best_val_loss,

            "runtime_per_epoch_minutes":
                results.runtime_per_epoch_minutes,
        }
    )

    run.finish()


def fail_wandb(run):
    if run is None:
        return

    run.finish(exit_code=1)