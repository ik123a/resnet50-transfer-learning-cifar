"""Run real notebook selection code with synthetic metrics, never train a model."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / 'resnet50_transfer_learning_cifar_colab.ipynb'


def code_cells():
    return [''.join(c.get('source', [])) for c in json.loads(NOTEBOOK.read_text())['cells'] if c['cell_type'] == 'code']


class ModelSelectionTests(unittest.TestCase):
    def test_configuration_ranking_uses_validation_not_test_accuracy(self):
        cell = next(s for s in code_cells() if 'comparison_df = pd.DataFrame' in s)
        namespace = {'pd': pd, 'results': {
            'test_winner': {'best_lr': 0.001, 'test_loss': 0.1, 'test_acc': 0.99, 'val_acc': 0.55},
            'validation_winner': {'best_lr': 0.001, 'test_loss': 0.4, 'test_acc': 0.70, 'val_acc': 0.85},
        }}
        exec(compile(cell, str(NOTEBOOK) + ':comparison', 'exec'), namespace)
        self.assertEqual('validation_winner', namespace['comparison_df'].iloc[0]['Model'])

    def test_experiment_reports_validation_of_the_loaded_checkpoint(self):
        source = next(s for s in code_cells() if s.startswith('def run_experiment('))
        model = SimpleNamespace(to=lambda device: model, parameters=lambda: [SimpleNamespace(requires_grad=True)], load_state_dict=lambda state: None)
        validation_loader, test_loader = object(), object()
        evaluated = []

        def evaluate(actual_model, loader, criterion, device):
            evaluated.append(loader)
            return (0.2, 0.85, [], []) if loader is validation_loader else (0.4, 0.70, [], [])

        namespace = {
            'tune_learning_rate': lambda **kwargs: 0.001,
            'build_resnet50_transfer': lambda *args, **kwargs: model,
            'print_layer_status': lambda model: None,
            'nn': SimpleNamespace(CrossEntropyLoss=lambda: None),
            'optim': SimpleNamespace(AdamW=lambda *args, **kwargs: None, lr_scheduler=SimpleNamespace(ReduceLROnPlateau=lambda *args, **kwargs: None)),
            'train_model': lambda **kwargs: (kwargs['model'], {}),
            'torch': SimpleNamespace(load=lambda *args, **kwargs: {}),
            'evaluate': evaluate,
            'os': __import__('os'),
            'LR_CANDIDATES': {'fully_frozen': [0.001]},
            'WEIGHT_DECAY': 0.0001, 'MODEL_SAVE_DIR': '.', 'DATASET_NAME': 'CIFAR10',
            'FAST_RUN': True, 'MAX_EPOCHS': 1, 'EARLY_STOPPING_PATIENCE': 1,
        }
        exec(compile(source, str(NOTEBOOK) + ':run_experiment', 'exec'), namespace)
        with redirect_stdout(StringIO()):
            result = namespace['run_experiment']('fully_frozen', object(), validation_loader, test_loader, 10, [], 'cpu')
        self.assertIn('val_acc', result)
        self.assertEqual(0.85, result['val_acc'])
        self.assertEqual(0.70, result['test_acc'])
        self.assertEqual([validation_loader, test_loader], evaluated)

    def test_python_cells_remain_syntactically_valid(self):
        for index, source in enumerate(code_cells()):
            if source.lstrip().startswith('!'):
                continue  # Notebook installation magic, not Python source.
            ast.parse(source, filename=f'{NOTEBOOK}:code-cell-{index}')


if __name__ == '__main__':
    unittest.main()
