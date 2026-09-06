import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import optuna
import yaml

ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec('wandb') is None:
    wandb = types.ModuleType('wandb')
    wandb.Api = object
    wandb.Artifact = object
    wandb.init = lambda *args, **kwargs: None
    sys.modules['wandb'] = wandb
train = types.ModuleType('train')
train.load_config = lambda path: yaml.safe_load(Path(path).read_text(encoding='utf-8'))
train.run_training = None
sys.modules['train'] = train
spec = importlib.util.spec_from_file_location('rxrx1_hpo_script', ROOT / 'scripts' / 'hpo.py')
hpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpo)


def metric_config():
    return {
        'experiment': {'name': 'metric_test', 'seed': 0},
        'optimizer': {
            'name': 'adamw',
            'base_lr': hpo.REFERENCE_BASE_LR,
            'lr_ratio': [1.0, hpo.FIXED_MIDDLE_RATIO, hpo.FIXED_LATE_RATIO, hpo.REFERENCE_HEAD_RATIO],
            'weight_decay': hpo.FIXED_WEIGHT_DECAY,
            'fusion_lr_ratio': hpo.FIXED_FUSION_LR_RATIO,
        },
        'model': {'dropout': hpo.FIXED_DROPOUT},
        'metric': {
            'enabled': True,
            'lambda_metric': 0.1,
            'wt': 0.7,
            'wc': 0.3,
            'alpha': 0.8,
            'min_pairs': 8,
            'projection_dims': [512, 128],
        },
        'training': {'epochs': 20},
        'scheduler': {'name': 'cosine', 'warmup_ratio': 0.05, 'min_lr_ratio': 0.01},
    }


class MetricHPOTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            'lambda_metric': 0.03,
            'wt': 0.62,
            'alpha': 0.91,
            'base_lr': 8.0e-5,
            'head_ratio': 34.0,
        }

    def test_metric_trial_params_and_fixed_values(self):
        config = metric_config()
        original_scheduler = dict(config['scheduler'])
        hpo.apply_trial_params(config, self.params)
        self.assertAlmostEqual(config['metric']['wc'], 1.0 - self.params['wt'])
        self.assertEqual(config['metric']['lambda_metric'], self.params['lambda_metric'])
        self.assertEqual(config['metric']['wt'], self.params['wt'])
        self.assertEqual(config['metric']['alpha'], self.params['alpha'])
        self.assertEqual(config['metric']['min_pairs'], 8)
        self.assertEqual(config['metric']['projection_dims'], [512, 128])
        self.assertEqual(config['optimizer']['base_lr'], self.params['base_lr'])
        self.assertEqual(config['optimizer']['lr_ratio'][1], hpo.FIXED_MIDDLE_RATIO)
        self.assertEqual(config['optimizer']['lr_ratio'][2], hpo.FIXED_LATE_RATIO)
        self.assertEqual(config['optimizer']['lr_ratio'][3], self.params['head_ratio'])
        self.assertEqual(config['optimizer']['weight_decay'], hpo.FIXED_WEIGHT_DECAY)
        self.assertEqual(config['optimizer']['fusion_lr_ratio'], hpo.FIXED_FUSION_LR_RATIO)
        self.assertEqual(config['model']['dropout'], hpo.FIXED_DROPOUT)
        self.assertEqual(config['scheduler'], original_scheduler)

    def test_metric_search_space_and_version(self):
        distributions = hpo.build_expected_distributions(SimpleNamespace(), 'metric')
        self.assertEqual(set(distributions), {'lambda_metric', 'wt', 'alpha', 'base_lr', 'head_ratio'})
        self.assertTrue(distributions['lambda_metric'].log)
        self.assertEqual(distributions['lambda_metric'].low, 0.01)
        self.assertEqual(distributions['lambda_metric'].high, 0.3)
        self.assertEqual(hpo.HPO_SPACE_VERSION, 'hierarchical-metric-v1')
        study = optuna.create_study()
        study.set_user_attr('hpo_space_version', hpo.CLASSIFICATION_HPO_SPACE_VERSION)
        with self.assertRaises(ValueError):
            hpo.validate_study_space(study, distributions, hpo.HPO_SPACE_VERSION)

    def test_resume_does_not_duplicate_enqueued_trials(self):
        study = optuna.create_study()
        self.assertEqual(hpo.enqueue_initial_trials(study, 'metric'), 3)
        first = [trial.system_attrs['fixed_params'] for trial in study.trials]
        self.assertEqual(hpo.enqueue_initial_trials(study, 'metric'), 0)
        self.assertEqual(len(study.trials), 3)
        self.assertEqual([trial.system_attrs['fixed_params'] for trial in study.trials], first)
        for trial in study.trials:
            fixed = trial.system_attrs['fixed_params']
            self.assertNotIn('wc', fixed)
            self.assertEqual(set(fixed), {'lambda_metric', 'wt', 'alpha', 'base_lr', 'head_ratio'})

    def test_best_config_exports_all_metric_hpo_values(self):
        study = optuna.create_study(direction='maximize')
        distributions = hpo.build_expected_distributions(SimpleNamespace(), 'metric')
        trial = optuna.trial.create_trial(
            params=self.params,
            distributions=distributions,
            value=0.42,
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(trial)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hpo.write_study_outputs(
                study,
                metric_config(),
                root,
                root / 'study.db',
                root / 'best.pt',
                hpo_mode='metric',
            )
            best = yaml.safe_load((root / 'best_config.yaml').read_text(encoding='utf-8'))
        for key in ('lambda_metric', 'wt', 'wc', 'alpha'):
            self.assertIn(key, best['metric'])
            self.assertIn(key, best['hpo'])
        self.assertIn('base_lr', best['hpo'])
        self.assertIn('head_ratio', best['hpo'])
        self.assertEqual(best['optimizer']['base_lr'], self.params['base_lr'])
        self.assertEqual(best['optimizer']['lr_ratio'][3], self.params['head_ratio'])
        self.assertAlmostEqual(best['hpo']['wc'], 1.0 - self.params['wt'])

    def test_classification_apply_trial_params_is_preserved(self):
        config = {'optimizer': {}, 'model': {}, 'training': {}, 'scheduler': {}}
        params = {
            'base_lr': 8e-5,
            'middle_ratio': 3.0,
            'late_ratio': 10.0,
            'head_ratio': 30.0,
            'weight_decay': 3e-4,
            'dropout': 0.22,
        }
        hpo.apply_trial_params(config, params, 'classification')
        self.assertEqual(config['optimizer']['lr_ratio'], [1.0, 3.0, 10.0, 30.0])
        self.assertEqual(config['optimizer']['weight_decay'], 3e-4)
        self.assertEqual(config['model']['dropout'], 0.22)
        self.assertEqual(config['training']['epochs'], 20)
        self.assertEqual(config['scheduler']['name'], 'cosine')


if __name__ == '__main__':
    unittest.main()
