from __future__ import annotations
import argparse, copy, gc
from pathlib import Path
import optuna, torch
import hpo as common
from train import load_config, run_training

SPACE_VERSION = "film-fusion-lr-v1"
BOUNDS = (1.0, 40.0)
ANCHORS = (1.0, 3.0, 10.0, 30.64707646122206, 40.0)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--study-name", default=None)
    p.add_argument("--timeout-hours", type=float, default=10.5)
    p.add_argument("--max-trials", type=int, default=15)
    p.add_argument("--disable-wandb", action="store_true")
    return p.parse_args()

def validate_config(config):
    meta = (config.get("model") or {}).get("metadata") or {}
    if not meta.get("enabled") or str(meta.get("method", "")).lower() != "film":
        raise ValueError("Fusion HPO requires enabled FiLM metadata.")
    cell, well = bool(meta.get("cell_type")), bool(meta.get("well_position"))
    if cell == well:
        raise ValueError("Enable exactly one of cell_type/well_position.")
    if (config.get("metric") or {}).get("enabled", False):
        raise ValueError("Fusion HPO requires metric learning disabled.")
    return "celltype" if cell else "well"

def distribution():
    return {"fusion_lr_ratio": optuna.distributions.FloatDistribution(*BOUNDS, log=True)}

def apply_params(config, params):
    config["optimizer"]["fusion_lr_ratio"] = float(params["fusion_lr_ratio"])

def sample_params(trial):
    return {"fusion_lr_ratio": trial.suggest_float("fusion_lr_ratio", *BOUNDS, log=True)}

def effective_params(config, params):
    ratio = float(params["fusion_lr_ratio"])
    return {"fusion_lr_ratio": ratio, "fusion_lr": float(config["optimizer"]["base_lr"]) * ratio}

def enqueue_anchors(study):
    if study.trials:
        return 0
    for ratio in ANCHORS:
        study.enqueue_trial({"fusion_lr_ratio": ratio})
    return len(ANCHORS)

def write_outputs(study, base_config, study_dir, storage_path, best_checkpoint):
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    counts = {s.name.lower(): sum(t.state == s for t in study.trials) for s in optuna.trial.TrialState}
    summary = study_dir / "study_summary.yaml"
    if not complete:
        common.save_yaml(summary, {
            "study_name": study.study_name, "storage": str(storage_path),
            "trial_counts": counts, "best_trial": None, "best_value": None, "best_params": None,
        })
        return

    best = study.best_trial
    config = copy.deepcopy(base_config)
    apply_params(config, best.params)
    config["experiment"]["name"] = f"{base_config['experiment']['name']}_hpo-fusion-lr-best"
    params = effective_params(base_config, best.params)
    config["hpo"] = {
        "study_name": study.study_name, "source_trial": best.number,
        "objective": "best_val_acc", "objective_value": best.value, **params,
    }
    common.save_yaml(study_dir / "best_config.yaml", config)
    common.save_yaml(summary, {
        "study_name": study.study_name, "storage": str(storage_path),
        "trial_counts": counts, "completed_trials": len(complete),
        "best_trial": best.number, "best_value": best.value, "best_params": params,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint.is_file() else None,
    })

def save_state(study, base_config, state_config, root, study_dir, db, sampler, best_pt):
    common.save_sampler(study, sampler)
    common.sync_best_checkpoint(study, best_pt)
    write_outputs(study, base_config, study_dir, db, best_pt)
    if state_config is None:
        return

    snap_dir, snap = study_dir / "_artifact_snapshot", study_dir / "_artifact_snapshot/study.db"
    common.create_database_snapshot(db, snap)
    try:
        common.upload_hpo_state(
            state_config=state_config, project_root=root, study=study, study_dir=study_dir,
            storage_snapshot_path=snap, sampler_path=sampler,
            best_config_path=study_dir / "best_config.yaml",
            summary_path=study_dir / "study_summary.yaml", best_checkpoint_path=best_pt,
        )
    finally:
        if snap.exists(): snap.unlink()
        if snap_dir.exists() and not any(snap_dir.iterdir()): snap_dir.rmdir()

def main():
    args = parse_args()
    if args.timeout_hours <= 0 or args.max_trials <= 0:
        raise ValueError("timeout-hours/max-trials must be > 0.")

    root = Path(__file__).resolve().parents[1]
    path = Path(args.config)
    path = path if path.is_absolute() else root / path
    base = load_config(path)
    kind = validate_config(base)
    args.study_name = args.study_name or f"{kind}_film_lr_hpo"

    slug = common.safe_name(args.study_name)
    study_dir = root / "outputs/hpo" / slug
    config_dir = study_dir / "configs"
    db, sampler, best_pt = (study_dir / "study.db").resolve(), study_dir / "sampler.pkl", study_dir / "best.pt"
    study_dir.mkdir(parents=True, exist_ok=True)

    state = common.get_wandb_state_config(base, args, slug)
    common.restore_hpo_state(state, study_dir, db, sampler)
    optuna_sampler = common.load_sampler(sampler)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=7, interval_steps=1)

    study = optuna.create_study(
        study_name=args.study_name, storage=f"sqlite:///{db.as_posix()}",
        direction="maximize", sampler=optuna_sampler, pruner=pruner, load_if_exists=True,
    )
    common.validate_study_space(study, distribution(), f"{kind}-{SPACE_VERSION}")
    if n := enqueue_anchors(study):
        print(f"Enqueued anchors: {n}")

    used = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    remaining = max(0, args.max_trials - used)
    print(f"Fusion type: {kind} | Study: {study.study_name} | Existing: {len(study.trials)} | Remaining: {remaining}")

    def objective(trial):
        config = copy.deepcopy(base)
        params = sample_params(trial)
        apply_params(config, params)
        name = f"{slug}_trial-{trial.number:04d}"
        config["experiment"]["name"] = name
        config["hpo"] = {"study_name": args.study_name, "trial_number": trial.number, **effective_params(base, params)}
        if args.disable_wandb:
            config.setdefault("wandb", {})["enabled"] = False

        config_path = config_dir / f"trial-{trial.number:04d}.yaml"
        checkpoint = root / config["checkpoint"]["dir"] / name / "best.pt"
        common.save_yaml(config_path, config)
        trial.set_user_attr("config_path", str(config_path))
        trial.set_user_attr("checkpoint_path", str(checkpoint))

        pruned = False
        def report(epoch, train_loss, train_acc, val_loss, val_acc):
            nonlocal pruned
            trial.report(float(val_acc), step=epoch)
            pruned = trial.should_prune()
            return pruned

        try:
            result = run_training(config, epoch_callback=report)
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if pruned:
            raise optuna.TrialPruned()
        return float(result.best_val_acc)

    def callback(current_study, trial):
        save_state(current_study, base, state, root, study_dir, db, sampler, best_pt)

    try:
        study.optimize(
            objective, n_trials=remaining, timeout=args.timeout_hours * 3600,
            n_jobs=1, gc_after_trial=True, callbacks=[callback], show_progress_bar=True,
        )
    finally:
        if db.is_file():
            save_state(study, base, state, root, study_dir, db, sampler, best_pt)

    if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        best = study.best_trial
        print(f"Best trial: {best.number}")
        print(f"Best val_acc: {best.value:.6f}")
        print(f"Best params: {effective_params(base, best.params)}")
        print(f"Best config: {study_dir / 'best_config.yaml'}")

if __name__ == "__main__":
    main()