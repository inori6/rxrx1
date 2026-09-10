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
    raise SystemExit("Optuna is required for HPO. Install it with: pip install optuna") from exc

from train import load_config, run_training


RATIO_BOUNDS = {
    "middle_ratio": (2.0, 5.0),
    "late_ratio": (7.0, 14.0),
    "head_ratio": (20.0, 45.0),
}

CLASSIFICATION_HPO_SPACE_VERSION = "focused-layer-lrs-v1"
HPO_SPACE_VERSION="final-hierarchical-b4-focused-v2"

METRIC_BOUNDS={
    "base_lr":(4.5e-5,6.0e-5,True),
    "middle_ratio":(3.8,5.5,True),
    "late_ratio":(7.0,11.0,True),
    "head_ratio":(26.0,36.0,True),
    "neck_lr_ratio":(10.0,18.0,True),
    "fusion_lr_ratio":(1.8,3.5,True),
    "weight_decay":(1e-4,3e-4,True),
    "dropout":(0.25,0.38,False),
    "lambda_metric":(0.03,0.07,True),
    "wt":(0.60,1.00,False),
    "alpha":(0.70,1.00,False),
}



def parse_args():
    parser = argparse.ArgumentParser(description="Resume-safe Bayesian HPO for RxRx1.")
    parser.add_argument("--config", default="configs/model_baseline.yaml", help="Fixed baseline YAML config.")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--timeout-hours", type=float, default=10.0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--base-lr-min", type=float, default=6.5e-5)
    parser.add_argument("--base-lr-max", type=float, default=1.15e-4)
    parser.add_argument("--weight-decay-min", type=float, default=3e-4)
    parser.add_argument("--weight-decay-max", type=float, default=3e-4)
    parser.add_argument("--dropout-min", type=float, default=0.22)
    parser.add_argument("--dropout-max", type=float, default=0.22)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def safe_name(value):
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return cleaned.strip("_") or "hpo"


def save_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(value, f, sort_keys=False, allow_unicode=True)


def get_hpo_mode(config):
    return "metric" if (config.get("metric") or {}).get("enabled", False) else "classification"


def get_space_version(hpo_mode):
    return HPO_SPACE_VERSION if hpo_mode == "metric" else CLASSIFICATION_HPO_SPACE_VERSION


def resolve_study_name(value, hpo_mode):
    if value:
        return value
    return "final_hierarchical_b4_hpo" if hpo_mode == "metric" else "model_baseline_hpo_focused"


def resolve_max_trials(value, hpo_mode):
    return value if value is not None else (50 if hpo_mode == "metric" else 20)


def apply_trial_params(config, params, hpo_mode=None):
    hpo_mode = hpo_mode or get_hpo_mode(config)
    optimizer = config.setdefault("optimizer", {})
    optimizer["name"] = "adamw"
    optimizer.pop("lr", None)
    optimizer["base_lr"] = float(params["base_lr"])

    if hpo_mode == "metric":
        optimizer["lr_ratio"]=[1.0,float(params["middle_ratio"]),float(params["late_ratio"]),float(params["head_ratio"])]
        optimizer["neck_lr_ratio"] = float(params["neck_lr_ratio"])
        optimizer["fusion_lr_ratio"] = float(params["fusion_lr_ratio"])
        optimizer["weight_decay"] = float(params["weight_decay"])
        config.setdefault("model", {})["dropout"] = float(params["dropout"])

        metric = config.setdefault("metric", {})
        wt = float(params["wt"])
        metric.update({
            "enabled": True,
            "lambda_metric": float(params["lambda_metric"]),
            "wt": wt,
            "wc": 1.0 - wt,
            "alpha": float(params["alpha"]),
        })
        return

    optimizer["lr_ratio"] = [
        1.0,
        float(params["middle_ratio"]),
        float(params["late_ratio"]),
        float(params["head_ratio"]),
    ]
    optimizer["weight_decay"] = float(params["weight_decay"])
    config.setdefault("model", {})["dropout"] = float(params["dropout"])
    config.setdefault("training", {})["epochs"] = 20
    config["scheduler"] = {"name": "cosine", "warmup_ratio": 0.05, "min_lr_ratio": 0.01}


def get_group_lrs(params, hpo_mode="classification"):
    base_lr = float(params["base_lr"])
    if hpo_mode == "metric":
        head_lr = base_lr * float(params["head_ratio"])
        return {
            "early_lr": base_lr,
            "middle_lr":base_lr*float(params["middle_ratio"]),
            "late_lr":base_lr*float(params["late_ratio"]),
            "head_lr": head_lr,
            "projection_lr": head_lr,
            "neck_lr": base_lr * float(params["neck_lr_ratio"]),
            "fusion_lr": base_lr * float(params["fusion_lr_ratio"]),
        }

    return {
        "early_lr": base_lr,
        "middle_lr": base_lr * float(params["middle_ratio"]),
        "late_lr": base_lr * float(params["late_ratio"]),
        "head_lr": base_lr * float(params["head_ratio"]),
    }


def build_expected_distributions(args, hpo_mode):
    if hpo_mode == "metric":
        return {
            name: optuna.distributions.FloatDistribution(low, high, log=log)
            for name, (low, high, log) in METRIC_BOUNDS.items()
        }

    return {
        "base_lr": optuna.distributions.FloatDistribution(args.base_lr_min, args.base_lr_max, log=True),
        "weight_decay": optuna.distributions.FloatDistribution(
            args.weight_decay_min, args.weight_decay_max, log=True
        ),
        "dropout": optuna.distributions.FloatDistribution(args.dropout_min, args.dropout_max),
        **{
            name: optuna.distributions.FloatDistribution(low, high, log=True)
            for name, (low, high) in RATIO_BOUNDS.items()
        },
    }


def sample_trial_params(trial, args, hpo_mode):
    if hpo_mode == "metric":
        return {
            name: trial.suggest_float(name, low, high, log=log)
            for name, (low, high, log) in METRIC_BOUNDS.items()
        }

    return {
        "base_lr": trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True),
        **{
            name: trial.suggest_float(name, low, high, log=True)
            for name, (low, high) in RATIO_BOUNDS.items()
        },
        "weight_decay": trial.suggest_float(
            "weight_decay", args.weight_decay_min, args.weight_decay_max, log=True
        ),
        "dropout": trial.suggest_float("dropout", args.dropout_min, args.dropout_max),
    }


def validate_study_space(study, expected_distributions, space_version):
    existing_version = study.user_attrs.get("hpo_space_version")
    incompatible = (
        existing_version not in (None, space_version)
        or (study.trials and existing_version is None)
        or any(
            expected_distributions.get(name) != distribution
            for trial in study.trials
            for name, distribution in trial.distributions.items()
        )
    )
    if incompatible:
        raise ValueError("Existing study uses a different search space. Choose a new --study-name.")
    study.set_user_attr("hpo_space_version", space_version)


def enqueue_initial_trials(study,hpo_mode):
    if hpo_mode!="metric" or study.trials:return 0
    study.enqueue_trial({
        "base_lr": 5.138340850683981e-05,
        "middle_ratio": 4.684416978885584,
        "late_ratio": 8.6406378065851,
        "head_ratio": 31.514789628881243,
        "neck_lr_ratio": 13.624907820609392,
        "fusion_lr_ratio": 2.4891812249513174,
        "weight_decay": 0.00015883625393345446,
        "dropout": 0.2979206389969634,
        "lambda_metric": 0.04507083758472812,
        "wt": 0.7141292228460144,
        "alpha": 0.9425686126605016,
    })
    return 1


def get_effective_best_params(params, hpo_mode):
    effective = dict(params)
    if hpo_mode == "metric":
        effective.update({"wc":1.0-float(params["wt"])})
    return effective


def get_wandb_state_config(base_config, args, study_slug):
    if args.disable_wandb:
        return None

    config = base_config.get("wandb") or {}
    if not config.get("enabled", False):
        return None

    mode = str(config.get("mode", "online")).lower()
    if mode != "online":
        print(f"W&B HPO state persistence disabled: mode={mode}")
        return None

    project = config.get("project")
    if not project:
        raise ValueError("wandb.project is required for HPO state persistence.")

    api = wandb.Api()
    entity = config.get("entity") or api.default_entity
    if not entity:
        raise RuntimeError("Unable to determine W&B entity.")

    identity = f"{entity}/{project}/{study_slug}"
    run_id = "hpostate" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]

    return {
        "entity": entity,
        "project": project,
        "mode": mode,
        "dir": config.get("dir", "outputs"),
        "artifact_name": f"{study_slug}-state",
        "state_run_id": run_id,
    }


def restore_hpo_state(state_config, study_dir, storage_path, sampler_path):
    if state_config is None:
        return False
    if storage_path.is_file():
        print("Local HPO database found. Skipping W&B restore.")
        return False

    ref = (
        f"{state_config['entity']}/"
        f"{state_config['project']}/"
        f"{state_config['artifact_name']}:latest"
    )
    print(f"Checking W&B HPO state: {ref}")

    api = wandb.Api()
    if not api.artifact_exists(ref, type="hpo-state"):
        print("No previous W&B HPO state found. Starting a new study.")
        return False

    api.artifact(ref, type="hpo-state").download(root=str(study_dir))
    missing = [p for p in (storage_path, sampler_path) if not p.is_file()]
    if missing:
        raise RuntimeError(f"Downloaded HPO state is incomplete. Missing: {missing}")

    print("Restored HPO state from W&B.")
    print(f"Database: {storage_path}")
    print(f"Sampler : {sampler_path}")
    return True


def load_sampler(sampler_path):
    if sampler_path.is_file():
        with sampler_path.open("rb") as f:
            sampler = pickle.load(f)
        print(f"Restored Optuna sampler from: {sampler_path}")
        return sampler

    return optuna.samplers.TPESampler(seed=0,n_startup_trials=3,multivariate=True)


def save_sampler(study, sampler_path):
    sampler_path.parent.mkdir(parents=True, exist_ok=True)
    with sampler_path.open("wb") as f:
        pickle.dump(study.sampler, f)


def create_database_snapshot(storage_path, snapshot_path):
    if not storage_path.is_file():
        raise FileNotFoundError(f"Optuna database not found: {storage_path}")

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        snapshot_path.unlink()

    source = sqlite3.connect(str(storage_path))
    destination = sqlite3.connect(str(snapshot_path))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


def sync_best_checkpoint(study, best_checkpoint_path):
    try:
        best_trial = study.best_trial
    except ValueError:
        return False

    source_value = best_trial.user_attrs.get("checkpoint_path")
    if source_value:
        source = Path(source_value)
        if source.is_file():
            best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != best_checkpoint_path.resolve():
                shutil.copy2(source, best_checkpoint_path)
            print(f"Updated persistent best checkpoint: {best_checkpoint_path}")

    return best_checkpoint_path.is_file()


def write_study_outputs(study, base_config, study_dir, storage_path, best_checkpoint_path, hpo_mode=None):
    hpo_mode = hpo_mode or get_hpo_mode(base_config)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_counts = {
        state.name.lower(): sum(t.state == state for t in study.trials)
        for state in optuna.trial.TrialState
    }

    summary_path = study_dir / "study_summary.yaml"
    best_config_path = study_dir / "best_config.yaml"

    if not completed:
        save_yaml(summary_path, {
            "study_name": study.study_name,
            "storage": str(storage_path),
            "trial_counts": trial_counts,
            "best_trial": None,
            "best_value": None,
            "best_params": None,
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path.is_file() else None,
        })
        return None

    best_trial = study.best_trial
    best_config = copy.deepcopy(base_config)
    apply_trial_params(best_config, best_trial.params, hpo_mode)

    suffix = "hpo-metric-best" if hpo_mode == "metric" else "hpo-ratio-best"
    best_config["experiment"]["name"] = f"{base_config['experiment']['name']}_{suffix}"

    group_lrs = get_group_lrs(best_trial.params, hpo_mode)
    best_config["hpo"] = {
        "study_name": study.study_name,
        "source_trial": best_trial.number,
        "objective": "best_val_acc",
        "objective_value": best_trial.value,
        **group_lrs,
    }

    if hpo_mode == "metric":
        best_config["hpo"].update(get_effective_best_params(best_trial.params, hpo_mode))

    save_yaml(best_config_path, best_config)
    save_yaml(summary_path, {
        "study_name": study.study_name,
        "storage": str(storage_path),
        "trial_counts": trial_counts,
        "completed_trials": len(completed),
        "best_trial": best_trial.number,
        "best_value": best_trial.value,
        "best_params": get_effective_best_params(best_trial.params, hpo_mode),
        "best_group_lrs": group_lrs,
        "best_trial_config": best_trial.user_attrs.get("config_path"),
        "source_checkpoint": best_trial.user_attrs.get("checkpoint_path"),
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path.is_file() else None,
    })
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
        name=state_config["artifact_name"],
        type="hpo-state",
        metadata={"study_name": study.study_name, "trials": len(study.trials)},
    )
    artifact.add_file(local_path=str(storage_snapshot_path), name="study.db")
    artifact.add_file(local_path=str(sampler_path), name="sampler.pkl")

    if best_config_path.is_file():
        artifact.add_file(local_path=str(best_config_path), name="best_config.yaml")
    if summary_path.is_file():
        artifact.add_file(local_path=str(summary_path), name="study_summary.yaml")
    if best_checkpoint_path.is_file():
        artifact.add_file(local_path=str(best_checkpoint_path), name="best.pt")

    wandb_dir = project_root / state_config["dir"]
    wandb_dir.mkdir(parents=True, exist_ok=True)

    with wandb.init(
        project=state_config["project"],
        entity=state_config["entity"],
        name=f"{safe_name(study.study_name)}_state",
        job_type="hpo-state",
        mode="online",
        dir=str(wandb_dir),
        config={"study_name": study.study_name, "artifact_name": state_config["artifact_name"]},
    ) as run:
        run.log_artifact(artifact, aliases=["latest"])

    print(f"Uploaded HPO state to W&B: {state_config['artifact_name']}:latest")


def save_hpo_state(
    study,
    base_config,
    state_config,
    project_root,
    study_dir,
    storage_path,
    sampler_path,
    best_checkpoint_path,
    hpo_mode=None,
):
    save_sampler(study, sampler_path)
    sync_best_checkpoint(study, best_checkpoint_path)
    write_study_outputs(
        study,
        base_config,
        study_dir,
        storage_path,
        best_checkpoint_path,
        hpo_mode,
    )

    if state_config is None:
        return

    snapshot_dir = study_dir / "_artifact_snapshot"
    snapshot_path = snapshot_dir / "study.db"
    create_database_snapshot(storage_path, snapshot_path)

    try:
        upload_hpo_state(
            state_config,
            project_root,
            study,
            study_dir,
            snapshot_path,
            sampler_path,
            study_dir / "best_config.yaml",
            study_dir / "study_summary.yaml",
            best_checkpoint_path,
        )
    finally:
        if snapshot_path.exists():
            snapshot_path.unlink()
        if snapshot_dir.exists() and not any(snapshot_dir.iterdir()):
            snapshot_dir.rmdir()


def main():
    args = parse_args()

    if args.timeout_hours <= 0:
        raise ValueError("--timeout-hours must be greater than zero.")
    if not 0 < args.base_lr_min <= args.base_lr_max:
        raise ValueError("Expected 0 < --base-lr-min <= --base-lr-max.")
    if not 0 < args.weight_decay_min <= args.weight_decay_max:
        raise ValueError("Expected 0 < --weight-decay-min <= --weight-decay-max.")
    if not 0 <= args.dropout_min <= args.dropout_max < 1:
        raise ValueError("Expected 0 <= dropout-min <= dropout-max < 1.")

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    base_config = load_config(config_path)
    hpo_mode = get_hpo_mode(base_config)

    args.study_name = resolve_study_name(args.study_name, hpo_mode)
    args.max_trials = resolve_max_trials(args.max_trials, hpo_mode)
    if args.max_trials <= 0:
        raise ValueError("--max-trials must be greater than zero.")

    study_slug = safe_name(args.study_name)

    if hpo_mode == "classification" and study_slug == "model_baseline_hpo":
        raise ValueError("Stage 2 requires a new study name.")

    if hpo_mode == "metric" and study_slug in {
        "model_baseline_hpo",
        "model_baseline_hpo_focused",
        "hierarchical_metric_hpo",
    }:
        raise ValueError("Final metric HPO requires an isolated study name.")

    study_dir = project_root / "outputs" / "hpo" / study_slug
    config_dir = study_dir / "configs"
    storage_path = (study_dir / "study.db").resolve()
    sampler_path = study_dir / "sampler.pkl"
    best_checkpoint_path = study_dir / "best.pt"
    storage_url = f"sqlite:///{storage_path.as_posix()}"

    study_dir.mkdir(parents=True, exist_ok=True)

    state_config = get_wandb_state_config(base_config, args, study_slug)
    restore_hpo_state(state_config, study_dir, storage_path, sampler_path)

    sampler = load_sampler(sampler_path)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=4,
        n_warmup_steps=4,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    expected = build_expected_distributions(args, hpo_mode)
    validate_study_space(study, expected, get_space_version(hpo_mode))

    enqueued = enqueue_initial_trials(study, hpo_mode)
    if enqueued:
        print(f"Enqueued metric anchors: {enqueued}")

    budgeted = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    remaining_trials = max(0, args.max_trials - budgeted)

    print(f"HPO mode        : {hpo_mode}")
    print(f"Study name      : {study.study_name}")
    print(f"Existing trials : {len(study.trials)}")
    print(f"Max trials      : {args.max_trials}")
    print(f"Remaining trials: {remaining_trials}")
    print(f"Database        : {storage_path}")

    def objective(trial):
        trial_config = copy.deepcopy(base_config)
        params = sample_trial_params(trial, args, hpo_mode)
        apply_trial_params(trial_config, params, hpo_mode)

        base_experiment_name = base_config["experiment"]["name"]
        trial_name = f"{study_slug}_trial-{trial.number:04d}"
        trial_config["experiment"]["name"] = trial_name

        group_lrs = get_group_lrs(params, hpo_mode)
        trial_config["hpo"] = {
            "study_name": args.study_name,
            "trial_number": trial.number,
            "base_experiment": base_experiment_name,
            "objective": "best_val_acc",
            **group_lrs,
        }

        if hpo_mode == "metric":
            trial_config["hpo"].update(get_effective_best_params(params, hpo_mode))
        if args.disable_wandb:
            trial_config.setdefault("wandb", {})["enabled"] = False

        trial_config_path = config_dir / f"trial-{trial.number:04d}.yaml"
        checkpoint_path = (
            project_root
            / trial_config["checkpoint"]["dir"]
            / trial_name
            / "best.pt"
        )

        save_yaml(trial_config_path, trial_config)

        trial.set_user_attr("experiment_name", trial_name)
        trial.set_user_attr("config_path", str(trial_config_path))
        trial.set_user_attr("checkpoint_path", str(checkpoint_path))
        for key, value in group_lrs.items():
            trial.set_user_attr(key, value)

        was_pruned = False

        def report_epoch(epoch_number, train_loss, train_acc, val_loss, val_acc):
            nonlocal was_pruned
            del train_loss, train_acc, val_loss
            trial.report(float(val_acc), step=epoch_number)
            if trial.should_prune():
                was_pruned = True
                return True
            return False

        try:
            results = run_training(trial_config, epoch_callback=report_epoch)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if was_pruned:
            raise optuna.TrialPruned()

        value = float(results.best_val_acc)
        trial.set_user_attr("best_val_acc", value)
        trial.set_user_attr("best_epoch", int(results.best_epoch))
        trial.set_user_attr("final_val_acc", float(results.final_val_acc))
        return value

    def persistence_callback(current_study, frozen_trial):
        print(f"Saving HPO state after trial {frozen_trial.number} ({frozen_trial.state.name})...")
        save_hpo_state(
            current_study,
            base_config,
            state_config,
            project_root,
            study_dir,
            storage_path,
            sampler_path,
            best_checkpoint_path,
            hpo_mode,
        )

    try:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=args.timeout_hours * 3600,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[persistence_callback],
        )
    finally:
        if storage_path.is_file():
            print("Saving final HPO state...")
            save_hpo_state(
                study,
                base_config,
                state_config,
                project_root,
                study_dir,
                storage_path,
                sampler_path,
                best_checkpoint_path,
                hpo_mode,
            )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed:
        print("No completed trials yet.")
        print(f"Resume database: {storage_path}")
        return

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best val accuracy: {best_trial.value:.6f}")
    print(f"Best parameters: {get_effective_best_params(best_trial.params, hpo_mode)}")
    print(f"Best config: {study_dir / 'best_config.yaml'}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Resume database: {storage_path}")

    if state_config is not None:
        print(f"W&B state artifact: {state_config['artifact_name']}:latest")


if __name__ == "__main__":
    main()