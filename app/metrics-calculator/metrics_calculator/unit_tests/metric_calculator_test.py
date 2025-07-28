import pytest
import pandas as pd
import numpy as np

from metrics_calculator.app import calculate_metrics
import metrics_calculator.logger_utils.exceptions as exceptions


@pytest.mark.parametrize("dataset, real, predicted, expected",
                         [('train',
                           pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                           pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                           {'rmse': 0.0, 'r2': 1.0, 'mape': 0.0, 'mse': 0.0}),
                          ('val',
                           pd.Series([1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
                           pd.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                           {'rmse': 1.0, 'r2': 0.0, 'mape': 1.0, 'mse': 1.0}),
                          ('test',
                           pd.Series([1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
                           pd.Series([1, 1, 1, 1, 1, 1, 0.5, 0.5], dtype=float),
                           {'rmse': 0.25, 'r2': 0.0, 'mape': 0.125, 'mse': 0.0625})])
def test_regression_metrics(dataset, real, predicted, expected):
    data = {
        dataset: {
            'y_real': real,
            'y_pred': predicted
        }
    }

    metadata = {
        'problem_type': 'regression',
        'metrics': ('rmse', 'r2', 'mape', 'mse'),
        'results_file_name': 'metrics_regression',
        'results_structure': 'single_row',
        'comment': 'Komentarz testowy',
        'save': False
    }

    results = calculate_metrics(data=data, metadata=metadata)

    for metric in ['rmse', 'r2', 'mape', 'mse']:
        assert results[f'{metric}_{dataset}'].values[0] == expected[metric], \
            f'Niepoprawna wartość metryki {metric}'


@pytest.mark.parametrize("dataset, real, predicted, expected",
                         [('train',
                           pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                           pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                           {'acc': 1.0, 'b_acc': 1.0, 'recall': 1.0,
                            'f1': 1.0, 'precision': 1.0}),
                          ('val',
                           pd.Series([1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
                           pd.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                           {'acc': 0.0, 'b_acc': 0.0, 'recall': 0.0,
                            'f1': 0.0, 'precision': 0.0}),
                          ('test',
                           pd.Series([1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
                           pd.Series([1, 1, 1, 1, 0, 0, 0, 0], dtype=float),
                           {'acc': 0.5, 'b_acc': 0.5, 'recall': 0.25,
                            'f1': 0.666667, 'precision': 1.0})])
def test_classification_metrics(dataset, real, predicted, expected):
    data = {
        dataset: {
            'y_real': real,
            'y_pred': predicted
        }
    }

    metadata = {
        'problem_type': 'classification',
        'metrics': ('acc', 'b_acc', 'recall', 'f1', 'precision'),
        'results_file_name': 'metrics_classification',
        'results_structure': 'single_row',
        'comment': 'Komentarz testowy',
        'save': False
    }

    results = calculate_metrics(data=data, metadata=metadata)

    for metric in ['acc', 'b_acc', 'recall', 'f1', 'precision']:
        assert round(results[f'{metric}_{dataset}'].values[0], 3) == \
            round(expected[metric], 3), f'Niepoprawna wartość metryki {metric}'


def test_ProblemTypeNotImplemented():
    with pytest.raises(exceptions.ProblemTypeNotImplemented):
        data = {
            'train': {
                'y_real': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                'y_pred': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
            }
        }

        metadata = {
            'problem_type': 'wrong_problem_type',
            'metrics': ('rmse', 'r2', 'mape', 'mse'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'single_row',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)


def test_MetricNotImplemented():
    with pytest.raises(exceptions.MetricNotImplemented):
        data = {
            'train': {
                'y_real': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                'y_pred': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
            }
        }

        metadata = {
            'problem_type': 'regression',
            'metrics': ('rmse', 'r2', 'mape', 'mse', 'wrong_metric'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'single_row',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)


def test_EmptyArray():
    with pytest.raises(exceptions.EmptyArray):
        data = {
            'train': {
                'y_real': pd.Series([], dtype=float),
                'y_pred': pd.Series([], dtype=float)
            }
        }

        metadata = {
            'problem_type': 'regression',
            'metrics': ('rmse', 'r2', 'mape', 'mse'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'single_row',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)


def test_IncompatibleLengths():
    with pytest.raises(exceptions.IncompatibleLengths):
        data = {
            'train': {
                'y_real': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                'y_pred': pd.Series([0, 0, 0, 0, 1, 1, 1], dtype=float)
            }
        }

        metadata = {
            'problem_type': 'regression',
            'metrics': ('rmse', 'r2', 'mape', 'mse'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'single_row',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)


def test_NanValuesDetected():
    with pytest.raises(exceptions.NanValuesDetected):
        data = {
            'train': {
                'y_real': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                'y_pred': pd.Series([0, 0, 0, 0, 1, 1, 1, np.nan], dtype=float)
            }
        }

        metadata = {
            'problem_type': 'regression',
            'metrics': ('rmse', 'r2', 'mape', 'mse'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'single_row',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)


def test_WrongType():
    with pytest.raises(exceptions.WrongType):
        data = {
            'train': {
                'y_real': [0, 0, 0, 0, 1, 1, 1, 1],
                'y_pred': [0, 0, 0, 0, 1, 1, 1, 1]
            }
        }

        metadata = {
            'problem_type': 'regression',
            'metrics': ('rmse', 'r2', 'mape', 'mse'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'single_row',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)


def test_WrongValuesType():
    with pytest.raises(exceptions.WrongValuesType):
        data = {
            'train': {
                'y_real': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=int),
                'y_pred': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
            }
        }

        metadata = {
            'problem_type': 'regression',
            'metrics': ('rmse', 'r2', 'mape', 'mse'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'single_row',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)


def test_IndexesNotMatch():
    with pytest.raises(exceptions.IndexesNotMatch):
        data = {
            'train': {
                'y_real': pd.Series([0, 0, 0, 0, 1, 1, 1, 1],
                                    index=[1, 2, 3, 4, 5, 6, 7, 8],
                                    dtype=float),
                'y_pred': pd.Series([0, 0, 0, 0, 1, 1, 1, 1],
                                    index=[0, 1, 2, 3, 4, 5, 6, 7],
                                    dtype=float)
            }
        }

        metadata = {
            'problem_type': 'regression',
            'metrics': ('rmse', 'r2', 'mape', 'mse'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'single_row',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)


def test_WrongDatasetName():
    with pytest.raises(exceptions.WrongDatasetName):
        data = {
            'wrong_name': {
                'y_real': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                'y_pred': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
            }
        }

        metadata = {
            'problem_type': 'regression',
            'metrics': ('rmse', 'r2', 'mape', 'mse'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'single_row',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)


def test_WrongResultsStructureName():
    with pytest.raises(exceptions.WrongResultsStructureName):
        data = {
            'train': {
                'y_real': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
                'y_pred': pd.Series([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
            }
        }

        metadata = {
            'problem_type': 'regression',
            'metrics': ('rmse', 'r2', 'mape', 'mse'),
            'results_file_name': 'metrics_regression',
            'results_structure': 'wrong_structure',
            'comment': 'Komentarz testowy',
            'save': False
        }

        calculate_metrics(data=data, metadata=metadata)
