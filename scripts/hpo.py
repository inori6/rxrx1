from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import pickle
import shutil
import sqlite3
from pathlib import Path

import torch
import wandb
import yaml

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "Optuna is required for HPO. "
        "Install it with: pip install optuna"
    ) from exc

from train import load_config, run_training


LR_RATIO_CHOICES = [
    "1_1_3_10",
    "1_3_5_10",
    "1_3_10_30",
]

SCHEDULER_CHOICES = [
    "cosine",
    "onecycle",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe Bayesian HPO "
            "for EfficientNet-B2."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/model_baseline.yaml",
        help="Fixed baseline YAML config.",
    )

    parser.add_argument(
        "--study-name",
        default="model_baseline_hpo",
    )

    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--base-lr-min",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--base-lr-max",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay-min",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--weight-decay-max",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--dropout-min",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--dropout-max",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--disable-wandb",
        action="store_true",
    )

    return parser.parse_args()


def safe_name(value):
    cleaned = "".join(
        character
        if (
            character.isalnum()
            or character in "-_"
        )
        else "_"
        for character in value
    )

    return (
        cleaned.strip("_")
        or "hpo"
    )


def save_yaml(path, value):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            value,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def apply_trial_params(
    config,
    params,
):
    optimizer_config = config.setdefault(
        "optimizer",
        {},
    )

    optimizer_config["name"] = "adamw"

    optimizer_config.pop(
        "lr",
        None,
    )

    optimizer_config["base_lr"] = float(
        params["base_lr"]
    )

    optimizer_config["lr_ratio"] = (
        params["lr_ratio"]
    )

    optimizer_config["weight_decay"] = float(
        params["weight_decay"]
    )

    model_config = config.setdefault(
        "model",
        {},
    )

    model_config["dropout"] = float(
        params["dropout"]
    )

    scheduler_name = params["scheduler"]

    if scheduler_name == "cosine":
        config["scheduler"] = {
            "name": "cosine",
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.01,
        }

    elif scheduler_name == "onecycle":
        config["scheduler"] = {
            "name": "onecycle",
            "pct_start": 0.1,
            "div_factor": 10.0,
            "final_div_factor": 1000.0,
        }

    else:
        raise ValueError(
            "Unsupported scheduler: "
            f"{scheduler_name}"
        )


def get_group_lrs(
    base_lr,
    lr_ratio,
):
    ratios = {
        "1_1_3_10": (
            1,
            1,
            3,
            10,
        ),
        "1_3_5_10": (
            1,
            3,
            5,
            10,
        ),
        "1_3_10_30": (
            1,
            3,
            10,
            30,
        ),
    }[lr_ratio]

    return {
        "early_lr": (
            base_lr
            * ratios[0]
        ),
        "middle_lr": (
            base_lr
            * ratios[1]
        ),
        "late_lr": (
            base_lr
            * ratios[2]
        ),
        "head_lr": (
            base_lr
            * ratios[3]
        ),
    }


def get_wandb_state_config(
    base_config,
    args,
    study_slug,
):
    if args.disable_wandb:
        return None

    wandb_config = (
        base_config.get("wandb")
        or {}
    )

    if not wandb_config.get(
        "enabled",
        False,
    ):
        return None

    mode = str(
        wandb_config.get(
            "mode",
            "online",
        )
    ).lower()

    if mode != "online":
        print(
            "W&B HPO state persistence disabled: "
            f"mode={mode}"
        )
        return None

    project = wandb_config.get(
        "project"
    )

    if not project:
        raise ValueError(
            "wandb.project is required "
            "for HPO state persistence."
        )

    api = wandb.Api()

    entity = (
        wandb_config.get("entity")
        or api.default_entity
    )

    if not entity:
        raise RuntimeError(
            "Unable to determine "
            "W&B entity."
        )

    state_identity = (
        f"{entity}/"
        f"{project}/"
        f"{study_slug}"
    )

    state_run_id = (
        "hpostate"
        + hashlib.sha1(
            state_identity.encode(
                "utf-8"
            )
        ).hexdigest()[:16]
    )

    return {
        "entity": entity,
        "project": project,
        "mode": mode,
        "dir": wandb_config.get(
            "dir",
            "outputs",
        ),
        "artifact_name": (
            f"{study_slug}-state"
        ),
        "state_run_id": (
            state_run_id
        ),
    }


def restore_hpo_state(
    state_config,
    study_dir,
    storage_path,
    sampler_path,
):
    if state_config is None:
        return False

    if storage_path.is_file():
        print(
            "Local HPO database found. "
            "Skipping W&B restore."
        )
        return False

    artifact_ref = (
        f"{state_config['entity']}/"
        f"{state_config['project']}/"
        f"{state_config['artifact_name']}"
        ":latest"
    )

    print(
        "Checking W&B HPO state: "
        f"{artifact_ref}"
    )

    api = wandb.Api()

    exists = api.artifact_exists(
        artifact_ref,
        type="hpo-state",
    )

    if not exists:
        print(
            "No previous W&B HPO state found. "
            "Starting a new study."
        )
        return False

    artifact = api.artifact(
        artifact_ref,
        type="hpo-state",
    )

    artifact.download(
        root=str(study_dir),
    )

    required_paths = [
        storage_path,
        sampler_path,
    ]

    missing = [
        path
        for path in required_paths
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Downloaded HPO state is incomplete. "
            f"Missing: {missing}"
        )

    print(
        "Restored HPO state from W&B."
    )
    print(
        f"Database: {storage_path}"
    )
    print(
        f"Sampler : {sampler_path}"
    )

    return True


def load_sampler(
    sampler_path,
):
    if sampler_path.is_file():
        with sampler_path.open(
            "rb"
        ) as file:
            sampler = pickle.load(
                file
            )

        print(
            "Restored Optuna sampler "
            f"from: {sampler_path}"
        )

        return sampler

    return optuna.samplers.TPESampler(
        seed=0,
        n_startup_trials=5,
        multivariate=True,
    )


def save_sampler(
    study,
    sampler_path,
):
    sampler_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sampler_path.open(
        "wb"
    ) as file:
        pickle.dump(
            study.sampler,
            file,
        )


def create_database_snapshot(
    storage_path,
    snapshot_path,
):
    if not storage_path.is_file():
        raise FileNotFoundError(
            f"Optuna database not found: "
            f"{storage_path}"
        )

    snapshot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if snapshot_path.exists():
        snapshot_path.unlink()

    source = sqlite3.connect(
        str(storage_path)
    )

    destination = sqlite3.connect(
        str(snapshot_path)
    )

    try:
        source.backup(
            destination
        )

    finally:
        destination.close()
        source.close()


def sync_best_checkpoint(
    study,
    best_checkpoint_path,
):
    try:
        best_trial = study.best_trial
    except ValueError:
        return False

    source_value = (
        best_trial.user_attrs.get(
            "checkpoint_path"
        )
    )

    if source_value:
        source_path = Path(
            source_value
        )

        if source_path.is_file():
            best_checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if (
                source_path.resolve()
                != best_checkpoint_path.resolve()
            ):
                shutil.copy2(
                    source_path,
                    best_checkpoint_path,
                )

            print(
                "Updated persistent "
                "best checkpoint: "
                f"{best_checkpoint_path}"
            )

    return (
        best_checkpoint_path.is_file()
    )


def write_study_outputs(
    study,
    base_config,
    study_dir,
    storage_path,
    best_checkpoint_path,
):
    completed_trials = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.COMPLETE
        )
    ]

    trial_counts = {}

    for state in optuna.trial.TrialState:
        trial_counts[
            state.name.lower()
        ] = sum(
            trial.state == state
            for trial in study.trials
        )

    summary_path = (
        study_dir
        / "study_summary.yaml"
    )

    best_config_path = (
        study_dir
        / "best_config.yaml"
    )

    if not completed_trials:
        save_yaml(
            summary_path,
            {
                "study_name": (
                    study.study_name
                ),
                "storage": (
                    str(storage_path)
                ),
                "trial_counts": (
                    trial_counts
                ),
                "best_trial": None,
                "best_value": None,
                "best_params": None,
                "best_checkpoint": (
                    str(
                        best_checkpoint_path
                    )
                    if (
                        best_checkpoint_path
                        .is_file()
                    )
                    else None
                ),
            },
        )

        return None

    best_trial = study.best_trial

    best_config = copy.deepcopy(
        base_config
    )

    apply_trial_params(
        best_config,
        best_trial.params,
    )

    best_config[
        "experiment"
    ]["name"] = (
        f"{base_config['experiment']['name']}"
        "_hpo-best"
    )

    best_group_lrs = get_group_lrs(
        best_trial.params[
            "base_lr"
        ],
        best_trial.params[
            "lr_ratio"
        ],
    )

    best_config["hpo"] = {
        "study_name": (
            study.study_name
        ),
        "source_trial": (
            best_trial.number
        ),
        "objective": (
            "best_val_acc"
        ),
        "objective_value": (
            best_trial.value
        ),
        **best_group_lrs,
    }

    save_yaml(
        best_config_path,
        best_config,
    )

    save_yaml(
        summary_path,
        {
            "study_name": (
                study.study_name
            ),
            "storage": (
                str(storage_path)
            ),
            "trial_counts": (
                trial_counts
            ),
            "completed_trials": (
                len(
                    completed_trials
                )
            ),
            "best_trial": (
                best_trial.number
            ),
            "best_value": (
                best_trial.value
            ),
            "best_params": (
                best_trial.params
            ),
            "best_group_lrs": (
                best_group_lrs
            ),
            "best_trial_config": (
                best_trial.user_attrs.get(
                    "config_path"
                )
            ),
            "source_checkpoint": (
                best_trial.user_attrs.get(
                    "checkpoint_path"
                )
            ),
            "best_checkpoint": (
                str(
                    best_checkpoint_path
                )
                if (
                    best_checkpoint_path
                    .is_file()
                )
                else None
            ),
        },
    )

    return best_trial


def upload_hpo_state(
    state_config,
    project_root,
    study,
    study_dir,
    storage_snapshot_path,
    sampler_path,
    best_config_path,
    summary_path,
    best_checkpoint_path,
):
    if state_config is None:
        return

    artifact = wandb.Artifact(
        name=(
            state_config[
                "artifact_name"
            ]
        ),
        type="hpo-state",
        metadata={
            "study_name": (
                study.study_name
            ),
            "trials": (
                len(study.trials)
            ),
        },
    )

    artifact.add_file(
        local_path=str(
            storage_snapshot_path
        ),
        name="study.db",
    )

    artifact.add_file(
        local_path=str(
            sampler_path
        ),
        name="sampler.pkl",
    )

    if best_config_path.is_file():
        artifact.add_file(
            local_path=str(
                best_config_path
            ),
            name="best_config.yaml",
        )

    if summary_path.is_file():
        artifact.add_file(
            local_path=str(
                summary_path
            ),
            name="study_summary.yaml",
        )

    if best_checkpoint_path.is_file():
        artifact.add_file(
            local_path=str(
                best_checkpoint_path
            ),
            name="best.pt",
        )

    wandb_dir = (
        project_root
        / state_config["dir"]
    )

    wandb_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wandb.init(
        project=(
            state_config["project"]
        ),
        entity=(
            state_config["entity"]
        ),
        id=(
            state_config[
                "state_run_id"
            ]
        ),
        name=(
            f"{safe_name(study.study_name)}"
            "_state"
        ),
        job_type="hpo-state",
        resume="allow",
        mode="online",
        dir=str(wandb_dir),
        config={
            "study_name": (
                study.study_name
            ),
            "artifact_name": (
                state_config[
                    "artifact_name"
                ]
            ),
        },
    ) as run:
        run.log_artifact(
            artifact,
            aliases=["latest"],
        )

    print(
        "Uploaded HPO state to W&B: "
        f"{state_config['artifact_name']}"
        ":latest"
    )


def save_hpo_state(
    study,
    base_config,
    state_config,
    project_root,
    study_dir,
    storage_path,
    sampler_path,
    best_checkpoint_path,
):
    save_sampler(
        study,
        sampler_path,
    )

    sync_best_checkpoint(
        study,
        best_checkpoint_path,
    )

    write_study_outputs(
        study=study,
        base_config=base_config,
        study_dir=study_dir,
        storage_path=storage_path,
        best_checkpoint_path=(
            best_checkpoint_path
        ),
    )

    if state_config is None:
        return

    snapshot_dir = (
        study_dir
        / "_artifact_snapshot"
    )

    snapshot_path = (
        snapshot_dir
        / "study.db"
    )

    create_database_snapshot(
        storage_path,
        snapshot_path,
    )

    try:
        upload_hpo_state(
            state_config=(
                state_config
            ),
            project_root=(
                project_root
            ),
            study=study,
            study_dir=study_dir,
            storage_snapshot_path=(
                snapshot_path
            ),
            sampler_path=(
                sampler_path
            ),
            best_config_path=(
                study_dir
                / "best_config.yaml"
            ),
            summary_path=(
                study_dir
                / "study_summary.yaml"
            ),
            best_checkpoint_path=(
                best_checkpoint_path
            ),
        )

    finally:
        if snapshot_path.exists():
            snapshot_path.unlink()

        if (
            snapshot_dir.exists()
            and not any(
                snapshot_dir.iterdir()
            )
        ):
            snapshot_dir.rmdir()


def main():
    args = parse_args()

    if args.timeout_hours <= 0:
        raise ValueError(
            "--timeout-hours must "
            "be greater than zero."
        )

    if args.max_trials <= 0:
        raise ValueError(
            "--max-trials must "
            "be greater than zero."
        )

    if not (
        0
        < args.base_lr_min
        <= args.base_lr_max
    ):
        raise ValueError(
            "Expected "
            "0 < --base-lr-min "
            "<= --base-lr-max."
        )

    if not (
        0
        < args.weight_decay_min
        <= args.weight_decay_max
    ):
        raise ValueError(
            "Expected "
            "0 < --weight-decay-min "
            "<= --weight-decay-max."
        )

    if not (
        0
        <= args.dropout_min
        <= args.dropout_max
        < 1
    ):
        raise ValueError(
            "Expected "
            "0 <= dropout-min "
            "<= dropout-max < 1."
        )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    config_path = Path(
        args.config
    )

    if not config_path.is_absolute():
        config_path = (
            project_root
            / config_path
        )

    base_config = load_config(
        config_path
    )

    study_slug = safe_name(
        args.study_name
    )

    study_dir = (
        project_root
        / "outputs"
        / "hpo"
        / study_slug
    )

    config_dir = (
        study_dir
        / "configs"
    )

    storage_path = (
        study_dir
        / "study.db"
    ).resolve()

    sampler_path = (
        study_dir
        / "sampler.pkl"
    )

    best_checkpoint_path = (
        study_dir
        / "best.pt"
    )

    storage_url = (
        "sqlite:///"
        f"{storage_path.as_posix()}"
    )

    study_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_config = (
        get_wandb_state_config(
            base_config=base_config,
            args=args,
            study_slug=study_slug,
        )
    )

    restore_hpo_state(
        state_config=state_config,
        study_dir=study_dir,
        storage_path=storage_path,
        sampler_path=sampler_path,
    )

    sampler = load_sampler(
        sampler_path
    )

    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=7,
            interval_steps=1,
        )
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    print(
        f"Study name      : "
        f"{study.study_name}"
    )
    print(
        f"Existing trials : "
        f"{len(study.trials)}"
    )
    print(
        f"Database        : "
        f"{storage_path}"
    )

    def objective(
        trial,
    ):
        trial_config = copy.deepcopy(
            base_config
        )

        params = {
            "base_lr": (
                trial.suggest_float(
                    "base_lr",
                    args.base_lr_min,
                    args.base_lr_max,
                    log=True,
                )
            ),
            "lr_ratio": (
                trial.suggest_categorical(
                    "lr_ratio",
                    LR_RATIO_CHOICES,
                )
            ),
            "weight_decay": (
                trial.suggest_float(
                    "weight_decay",
                    args.weight_decay_min,
                    args.weight_decay_max,
                    log=True,
                )
            ),
            "dropout": (
                trial.suggest_float(
                    "dropout",
                    args.dropout_min,
                    args.dropout_max,
                )
            ),
            "scheduler": (
                trial.suggest_categorical(
                    "scheduler",
                    SCHEDULER_CHOICES,
                )
            ),
        }

        apply_trial_params(
            trial_config,
            params,
        )

        base_experiment_name = (
            base_config[
                "experiment"
            ]["name"]
        )

        trial_name = (
            f"{study_slug}_"
            f"trial-{trial.number:04d}"
        )

        trial_config[
            "experiment"
        ]["name"] = trial_name

        group_lrs = get_group_lrs(
            params["base_lr"],
            params["lr_ratio"],
        )

        trial_config["hpo"] = {
            "study_name": (
                args.study_name
            ),
            "trial_number": (
                trial.number
            ),
            "base_experiment": (
                base_experiment_name
            ),
            "objective": (
                "best_val_acc"
            ),
            **group_lrs,
        }

        if args.disable_wandb:
            trial_config.setdefault(
                "wandb",
                {},
            )["enabled"] = False

        trial_config_path = (
            config_dir
            / (
                f"trial-"
                f"{trial.number:04d}"
                f".yaml"
            )
        )

        checkpoint_path = (
            project_root
            / trial_config[
                "checkpoint"
            ]["dir"]
            / trial_name
            / "best.pt"
        )

        save_yaml(
            trial_config_path,
            trial_config,
        )

        trial.set_user_attr(
            "experiment_name",
            trial_name,
        )

        trial.set_user_attr(
            "config_path",
            str(
                trial_config_path
            ),
        )

        trial.set_user_attr(
            "checkpoint_path",
            str(
                checkpoint_path
            ),
        )

        for (
            key,
            value,
        ) in group_lrs.items():
            trial.set_user_attr(
                key,
                value,
            )

        was_pruned = False

        def report_epoch(
            epoch_number,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        ):
            nonlocal was_pruned

            del (
                train_loss,
                train_acc,
                val_loss,
            )

            trial.report(
                float(val_acc),
                step=epoch_number,
            )

            if trial.should_prune():
                was_pruned = True
                return True

            return False

        try:
            results = run_training(
                trial_config,
                epoch_callback=(
                    report_epoch
                ),
            )

        finally:
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if was_pruned:
            raise optuna.TrialPruned()

        objective_value = float(
            results.best_val_acc
        )

        trial.set_user_attr(
            "best_val_acc",
            objective_value,
        )

        trial.set_user_attr(
            "best_epoch",
            int(
                results.best_epoch
            ),
        )

        trial.set_user_attr(
            "final_val_acc",
            float(
                results.final_val_acc
            ),
        )

        return objective_value

    def persistence_callback(
        current_study,
        frozen_trial,
    ):
        print(
            "Saving HPO state after "
            f"trial {frozen_trial.number} "
            f"({frozen_trial.state.name})..."
        )

        save_hpo_state(
            study=current_study,
            base_config=base_config,
            state_config=state_config,
            project_root=project_root,
            study_dir=study_dir,
            storage_path=storage_path,
            sampler_path=sampler_path,
            best_checkpoint_path=(
                best_checkpoint_path
            ),
        )

    try:
        study.optimize(
            objective,
            n_trials=args.max_trials,
            timeout=(
                args.timeout_hours
                * 60
                * 60
            ),
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[
                persistence_callback
            ],
        )

    finally:
        if storage_path.is_file():
            print(
                "Saving final HPO state..."
            )

            save_hpo_state(
                study=study,
                base_config=base_config,
                state_config=state_config,
                project_root=project_root,
                study_dir=study_dir,
                storage_path=storage_path,
                sampler_path=sampler_path,
                best_checkpoint_path=(
                    best_checkpoint_path
                ),
            )

    completed_trials = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.COMPLETE
        )
    ]

    if not completed_trials:
        print(
            "No completed trials yet."
        )
        print(
            f"Resume database: "
            f"{storage_path}"
        )
        return

    best_trial = study.best_trial

    print(
        f"Best trial: "
        f"{best_trial.number}"
    )

    print(
        f"Best val accuracy: "
        f"{best_trial.value:.6f}"
    )

    print(
        f"Best parameters: "
        f"{best_trial.params}"
    )

    print(
        f"Best config: "
        f"{study_dir / 'best_config.yaml'}"
    )

    print(
        f"Best checkpoint: "
        f"{best_checkpoint_path}"
    )

    print(
        f"Resume database: "
        f"{storage_path}"
    )

    if state_config is not None:
        print(
            "W&B state artifact: "
            f"{state_config['artifact_name']}"
            ":latest"
        )


if __name__ == "__main__":
    main()