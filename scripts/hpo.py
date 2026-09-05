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
    raise SystemExit('Optuna is required for HPO. Install it with: pip install optuna') from exc
from train import load_config, run_training
RATIO_BOUNDS = {
    'middle_ratio': (2.0, 5.0),
    'late_ratio': (7.0, 14.0),
    'head_ratio': (20.0, 45.0),
}
HPO_SPACE_VERSION = 'focused-layer-lrs-v1'

def parse_args():
    parser = argparse.ArgumentParser(description='Resume-safe Bayesian HPO for EfficientNet-B2.')
    parser.add_argument(
        '--config',
        default='configs/model_baseline.yaml',
        help='Fixed baseline YAML config.',
    )
    parser.add_argument('--study-name', default='model_baseline_hpo_focused')
    parser.add_argument(
        '--timeout-hours',
        type=float,
        default=10.0,
        help='Per-invocation timeout; a running trial finishes before stopping.',
    )
    parser.add_argument(
        '--max-trials',
        type=int,
        default=20,
        help='Total trial budget, including pruned and failed trials.',
    )
    parser.add_argument('--base-lr-min', type=float, default=6.5e-05)
    parser.add_argument('--base-lr-max', type=float, default=1.15e-04)
    # Equal lower/upper bounds keep these parameters fixed while preserving the
    # existing config export and resume logic.
    parser.add_argument('--weight-decay-min', type=float, default=3e-04)
    parser.add_argument('--weight-decay-max', type=float, default=3e-04)
    parser.add_argument('--dropout-min', type=float, default=0.22)
    parser.add_argument('--dropout-max', type=float, default=0.22)
    parser.add_argument('--disable-wandb', action='store_true')
    return parser.parse_args()

def safe_name(value):
    cleaned = ''.join((character if character.isalnum() or character in '-_' else '_' for character in value))
    return cleaned.strip('_') or 'hpo'

def save_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(value, file, sort_keys=False, allow_unicode=True)

def apply_trial_params(config, params):
    optimizer_config = config.setdefault('optimizer', {})
    optimizer_config['name'] = 'adamw'
    optimizer_config.pop('lr', None)
    optimizer_config['base_lr'] = float(params['base_lr'])
    optimizer_config['lr_ratio'] = [
        1.0,
        float(params['middle_ratio']),
        float(params['late_ratio']),
        float(params['head_ratio']),
    ]
    optimizer_config['weight_decay'] = float(params['weight_decay'])
    model_config = config.setdefault('model', {})
    model_config['dropout'] = float(params['dropout'])
    config.setdefault('training', {})['epochs'] = 20
    config['scheduler'] = {'name': 'cosine', 'warmup_ratio': 0.05, 'min_lr_ratio': 0.01}

def get_group_lrs(params):
    base_lr = float(params['base_lr'])
    return {
        'early_lr': base_lr,
        'middle_lr': base_lr * float(params['middle_ratio']),
        'late_lr': base_lr * float(params['late_ratio']),
        'head_lr': base_lr * float(params['head_ratio']),
    }

def get_wandb_state_config(base_config, args, study_slug):
    if args.disable_wandb:
        return None
    wandb_config = base_config.get('wandb') or {}
    if not wandb_config.get('enabled', False):
        return None
    mode = str(wandb_config.get('mode', 'online')).lower()
    if mode != 'online':
        print(f'W&B HPO state persistence disabled: mode={mode}')
        return None
    project = wandb_config.get('project')
    if not project:
        raise ValueError('wandb.project is required for HPO state persistence.')
    api = wandb.Api()
    entity = wandb_config.get('entity') or api.default_entity
    if not entity:
        raise RuntimeError('Unable to determine W&B entity.')
    state_identity = f'{entity}/{project}/{study_slug}'
    state_run_id = 'hpostate' + hashlib.sha1(state_identity.encode('utf-8')).hexdigest()[:16]
    return {
        'entity': entity,
        'project': project,
        'mode': mode,
        'dir': wandb_config.get('dir', 'outputs'),
        'artifact_name': f'{study_slug}-state',
        'state_run_id': state_run_id,
    }

def restore_hpo_state(state_config, study_dir, storage_path, sampler_path):
    if state_config is None:
        return False
    if storage_path.is_file():
        print('Local HPO database found. Skipping W&B restore.')
        return False
    artifact_ref = f"{state_config['entity']}/{state_config['project']}/{state_config['artifact_name']}:latest"
    print(f'Checking W&B HPO state: {artifact_ref}')
    api = wandb.Api()
    exists = api.artifact_exists(artifact_ref, type='hpo-state')
    if not exists:
        print('No previous W&B HPO state found. Starting a new study.')
        return False
    artifact = api.artifact(artifact_ref, type='hpo-state')
    artifact.download(root=str(study_dir))
    required_paths = [storage_path, sampler_path]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f'Downloaded HPO state is incomplete. Missing: {missing}')
    print('Restored HPO state from W&B.')
    print(f'Database: {storage_path}')
    print(f'Sampler : {sampler_path}')
    return True

def load_sampler(sampler_path):
    if sampler_path.is_file():
        with sampler_path.open('rb') as file:
            sampler = pickle.load(file)
        print(f'Restored Optuna sampler from: {sampler_path}')
        return sampler
    return optuna.samplers.TPESampler(seed=0, n_startup_trials=5, multivariate=True)

def save_sampler(study, sampler_path):
    sampler_path.parent.mkdir(parents=True, exist_ok=True)
    with sampler_path.open('wb') as file:
        pickle.dump(study.sampler, file)

def create_database_snapshot(storage_path, snapshot_path):
    if not storage_path.is_file():
        raise FileNotFoundError(f'Optuna database not found: {storage_path}')
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
    source_value = best_trial.user_attrs.get('checkpoint_path')
    if source_value:
        source_path = Path(source_value)
        if source_path.is_file():
            best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != best_checkpoint_path.resolve():
                shutil.copy2(source_path, best_checkpoint_path)
            print(f'Updated persistent best checkpoint: {best_checkpoint_path}')
    return best_checkpoint_path.is_file()

def write_study_outputs(study, base_config, study_dir, storage_path, best_checkpoint_path):
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    trial_counts = {}
    for state in optuna.trial.TrialState:
        trial_counts[state.name.lower()] = sum((trial.state == state for trial in study.trials))
    summary_path = study_dir / 'study_summary.yaml'
    best_config_path = study_dir / 'best_config.yaml'
    if not completed_trials:
        save_yaml(
            summary_path,
            {
                'study_name': study.study_name,
                'storage': str(storage_path),
                'trial_counts': trial_counts,
                'best_trial': None,
                'best_value': None,
                'best_params': None,
                'best_checkpoint': str(best_checkpoint_path) if best_checkpoint_path.is_file() else None,
            },
        )
        return None
    best_trial = study.best_trial
    best_config = copy.deepcopy(base_config)
    apply_trial_params(best_config, best_trial.params)
    best_config['experiment']['name'] = f"{base_config['experiment']['name']}_hpo-ratio-best"
    best_group_lrs = get_group_lrs(best_trial.params)
    best_config['hpo'] = {
        'study_name': study.study_name,
        'source_trial': best_trial.number,
        'objective': 'best_val_acc',
        'objective_value': best_trial.value,
        **best_group_lrs,
    }
    save_yaml(best_config_path, best_config)
    save_yaml(
        summary_path,
        {
            'study_name': study.study_name,
            'storage': str(storage_path),
            'trial_counts': trial_counts,
            'completed_trials': len(completed_trials),
            'best_trial': best_trial.number,
            'best_value': best_trial.value,
            'best_params': best_trial.params,
            'best_group_lrs': best_group_lrs,
            'best_trial_config': best_trial.user_attrs.get('config_path'),
            'source_checkpoint': best_trial.user_attrs.get('checkpoint_path'),
            'best_checkpoint': str(best_checkpoint_path) if best_checkpoint_path.is_file() else None,
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
        name=state_config['artifact_name'],
        type='hpo-state',
        metadata={'study_name': study.study_name, 'trials': len(study.trials)},
    )
    artifact.add_file(local_path=str(storage_snapshot_path), name='study.db')
    artifact.add_file(local_path=str(sampler_path), name='sampler.pkl')
    if best_config_path.is_file():
        artifact.add_file(local_path=str(best_config_path), name='best_config.yaml')
    if summary_path.is_file():
        artifact.add_file(local_path=str(summary_path), name='study_summary.yaml')
    if best_checkpoint_path.is_file():
        artifact.add_file(local_path=str(best_checkpoint_path), name='best.pt')
    wandb_dir = project_root / state_config['dir']
    wandb_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        project=state_config['project'],
        entity=state_config['entity'],
        id=state_config['state_run_id'],
        name=f'{safe_name(study.study_name)}_state',
        job_type='hpo-state',
        resume='allow',
        mode='online',
        dir=str(wandb_dir),
        config={'study_name': study.study_name, 'artifact_name': state_config['artifact_name']},
    ) as run:
        run.log_artifact(artifact, aliases=['latest'])
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
):
    save_sampler(study, sampler_path)
    sync_best_checkpoint(study, best_checkpoint_path)
    write_study_outputs(
        study=study,
        base_config=base_config,
        study_dir=study_dir,
        storage_path=storage_path,
        best_checkpoint_path=best_checkpoint_path,
    )
    if state_config is None:
        return
    snapshot_dir = study_dir / '_artifact_snapshot'
    snapshot_path = snapshot_dir / 'study.db'
    create_database_snapshot(storage_path, snapshot_path)
    try:
        upload_hpo_state(
            state_config=state_config,
            project_root=project_root,
            study=study,
            study_dir=study_dir,
            storage_snapshot_path=snapshot_path,
            sampler_path=sampler_path,
            best_config_path=study_dir / 'best_config.yaml',
            summary_path=study_dir / 'study_summary.yaml',
            best_checkpoint_path=best_checkpoint_path,
        )
    finally:
        if snapshot_path.exists():
            snapshot_path.unlink()
        if snapshot_dir.exists() and (not any(snapshot_dir.iterdir())):
            snapshot_dir.rmdir()

def main():
    args = parse_args()
    if args.timeout_hours <= 0:
        raise ValueError('--timeout-hours must be greater than zero.')
    if args.max_trials <= 0:
        raise ValueError('--max-trials must be greater than zero.')
    if not 0 < args.base_lr_min <= args.base_lr_max:
        raise ValueError('Expected 0 < --base-lr-min <= --base-lr-max.')
    if not 0 < args.weight_decay_min <= args.weight_decay_max:
        raise ValueError('Expected 0 < --weight-decay-min <= --weight-decay-max.')
    if not 0 <= args.dropout_min <= args.dropout_max < 1:
        raise ValueError('Expected 0 <= dropout-min <= dropout-max < 1.')
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    base_config = load_config(config_path)
    study_slug = safe_name(args.study_name)
    if study_slug == 'model_baseline_hpo':
        raise ValueError('Stage 2 requires a new study name.')
    study_dir = project_root / 'outputs' / 'hpo' / study_slug
    config_dir = study_dir / 'configs'
    storage_path = (study_dir / 'study.db').resolve()
    sampler_path = study_dir / 'sampler.pkl'
    best_checkpoint_path = study_dir / 'best.pt'
    storage_url = f'sqlite:///{storage_path.as_posix()}'
    study_dir.mkdir(parents=True, exist_ok=True)
    state_config = get_wandb_state_config(base_config=base_config, args=args, study_slug=study_slug)
    restore_hpo_state(
        state_config=state_config,
        study_dir=study_dir,
        storage_path=storage_path,
        sampler_path=sampler_path,
    )
    sampler = load_sampler(sampler_path)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=7, interval_steps=1)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    # Resume only this search space, including partially sampled trials.
    expected_distributions = {
        'base_lr': optuna.distributions.FloatDistribution(
            args.base_lr_min,
            args.base_lr_max,
            log=True,
        ),
        'weight_decay': optuna.distributions.FloatDistribution(
            args.weight_decay_min,
            args.weight_decay_max,
            log=True,
        ),
        'dropout': optuna.distributions.FloatDistribution(args.dropout_min, args.dropout_max),
        **{name: optuna.distributions.FloatDistribution(
            low,
            high,
            log=True,
        ) for name, (low, high) in RATIO_BOUNDS.items()},
    }
    space_version = study.user_attrs.get('hpo_space_version')
    if (
        space_version not in (None, HPO_SPACE_VERSION)
        or (study.trials and space_version is None)
        or any(
            expected_distributions.get(name) != distribution
            for trial in study.trials
            for name, distribution in trial.distributions.items()
        )
    ):
        raise ValueError('Existing study uses a different search space. Choose a new --study-name.')
    study.set_user_attr('hpo_space_version', HPO_SPACE_VERSION)
    # The budget applies to the whole study, not each resumed invocation.
    remaining_trials = max(0, args.max_trials - len(study.trials))
    print(f'Study name      : {study.study_name}')
    print(f'Existing trials : {len(study.trials)}')
    print(f'Database        : {storage_path}')

    def objective(trial):
        trial_config = copy.deepcopy(base_config)
        params = {
            'base_lr': trial.suggest_float('base_lr', args.base_lr_min, args.base_lr_max, log=True),
            **{name: trial.suggest_float(
                name,
                low,
                high,
                log=True,
            ) for name, (low, high) in RATIO_BOUNDS.items()},
            'weight_decay': trial.suggest_float(
                'weight_decay',
                args.weight_decay_min,
                args.weight_decay_max,
                log=True,
            ),
            'dropout': trial.suggest_float('dropout', args.dropout_min, args.dropout_max),
        }
        apply_trial_params(trial_config, params)
        base_experiment_name = base_config['experiment']['name']
        trial_name = f'{study_slug}_trial-{trial.number:04d}'
        trial_config['experiment']['name'] = trial_name
        group_lrs = get_group_lrs(params)
        trial_config['hpo'] = {
            'study_name': args.study_name,
            'trial_number': trial.number,
            'base_experiment': base_experiment_name,
            'objective': 'best_val_acc',
            **group_lrs,
        }
        if args.disable_wandb:
            trial_config.setdefault('wandb', {})['enabled'] = False
        trial_config_path = config_dir / f'trial-{trial.number:04d}.yaml'
        checkpoint_path = project_root / trial_config['checkpoint']['dir'] / trial_name / 'best.pt'
        save_yaml(trial_config_path, trial_config)
        trial.set_user_attr('experiment_name', trial_name)
        trial.set_user_attr('config_path', str(trial_config_path))
        trial.set_user_attr('checkpoint_path', str(checkpoint_path))
        for key, value in group_lrs.items():
            trial.set_user_attr(key, value)
        was_pruned = False

        def report_epoch(epoch_number, train_loss, train_acc, val_loss, val_acc):
            nonlocal was_pruned
            del (train_loss, train_acc, val_loss)
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
        objective_value = float(results.best_val_acc)
        trial.set_user_attr('best_val_acc', objective_value)
        trial.set_user_attr('best_epoch', int(results.best_epoch))
        trial.set_user_attr('final_val_acc', float(results.final_val_acc))
        return objective_value

    def persistence_callback(current_study, frozen_trial):
        print(f'Saving HPO state after trial {frozen_trial.number} ({frozen_trial.state.name})...')
        save_hpo_state(
            study=current_study,
            base_config=base_config,
            state_config=state_config,
            project_root=project_root,
            study_dir=study_dir,
            storage_path=storage_path,
            sampler_path=sampler_path,
            best_checkpoint_path=best_checkpoint_path,
        )
    try:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=args.timeout_hours * 60 * 60,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[persistence_callback],
        )
    finally:
        if storage_path.is_file():
            print('Saving final HPO state...')
            save_hpo_state(
                study=study,
                base_config=base_config,
                state_config=state_config,
                project_root=project_root,
                study_dir=study_dir,
                storage_path=storage_path,
                sampler_path=sampler_path,
                best_checkpoint_path=best_checkpoint_path,
            )
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print('No completed trials yet.')
        print(f'Resume database: {storage_path}')
        return
    best_trial = study.best_trial
    print(f'Best trial: {best_trial.number}')
    print(f'Best val accuracy: {best_trial.value:.6f}')
    print(f'Best parameters: {best_trial.params}')
    print(f"Best config: {study_dir / 'best_config.yaml'}")
    print(f'Best checkpoint: {best_checkpoint_path}')
    print(f'Resume database: {storage_path}')
    if state_config is not None:
        print(f"W&B state artifact: {state_config['artifact_name']}:latest")

if __name__ == '__main__':
    main()