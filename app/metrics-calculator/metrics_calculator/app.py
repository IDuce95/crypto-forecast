import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_percentage_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)

from metrics_calculator.validation import (run_validation,
                                           validate_metadata_dict)


def calculate_metrics(
    data: dict,
    metadata: dict,
) -> pd.DataFrame:
    """ Zadaniem tej funkcji jest zwrócenie obliczonych metryk dla zadanych
        danych wejściowych. Funkcja przyjmuje słownik z obiektami typu Series,
        dla których obliczy metryki oraz słownik z metadanymi:
        - problem_type: rodzaj problemu (regression/classification),
        - metrics: krotka z metrykami, które zostaną obliczone,
        - results_file_name: nazwa pliku, w którym zapisane zostaną wyniki,
        - results_structure: rodzaj struktury zwracanego obiektu,
        - comment: komentarz dla wyników,
        - save: wartość typu bool, określająca czy zapisać wyniki do pliku

        Funkcja oblicza zadane metryki dla każdego zbioru (train/val/test)
        z osobna i generuje obiekt typu DataFrame o zadanej strukturze
        (definiowanej zmienną results_structure) i je zwraca oraz
        opcjonalnie zapisuje wyniki do pliku o zadanej nazwie. Można również
        zdefiniować pole 'comment' w słowniku metadata, natomiast komentarz ten
        zostanie uwzględniony jedynie kiedy pole 'results_structure' będzie
        równe 'single_row'. Jeśli 'results_structure' przyjmie 'multiple_rows'
        to komentarz zostanie pominięty.

    Args:
        data (dict): słownik z danymi, dla których obliczone zostaną metryki
        metadata (dict): słownik z metadanymi, definiującymi aktualny przypadek

    Returns:
        pd.DataFrame: obiekt, zawierający metryki
    """

    problem_type, metrics, results_file_name, \
        results_structure, comment, save = unpack_metadata(metadata)
    metrics_dict = get_metrics_dict()

    results = pd.DataFrame()

    for dataset in data.keys():
        real = data[dataset]['y_real']
        predicted = data[dataset]['y_pred']
        run_validation(real, predicted, metrics, metrics_dict,
                       problem_type, dataset, results_structure)

        results = get_results(results, real, predicted, metrics_dict,
                              problem_type, dataset, comment,
                              metrics, results_structure)
    if save:
        save_results(results, results_file_name)
    return results


def get_results(
    results: pd.DataFrame,
    real: pd.Series,
    predicted: pd.Series,
    metrics_dict: dict,
    problem_type: str,
    dataset: str,
    comment: str,
    metrics: Tuple[str, ...],
    results_structure: str
) -> pd.DataFrame:

    if results_structure == 'multiple_rows':
        for metric in metrics:
            results.at[dataset, metric] = \
                metrics_dict[problem_type][metric](real, predicted)

    elif results_structure == 'single_row':
        results.at[0, 'comment'] = comment
        for metric in metrics:
            results.at[0, f'{metric}_{dataset}'] = \
                metrics_dict[problem_type][metric](real, predicted)

    return results


def unpack_metadata(metadata: dict) -> Tuple[str, ...]:

    validate_metadata_dict(metadata)

    problem_type = metadata['problem_type']
    metrics = metadata['metrics']
    results_file_name = metadata['results_file_name']
    results_structure = metadata['results_structure']
    comment = metadata['comment']
    save = metadata['save']

    return problem_type, metrics, results_file_name, \
        results_structure, comment, save


def get_metrics_dict() -> dict:
    metrics_dict = {
        'regression': {
            'mape': mape,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        },
        'classification': {
            'acc': accuracy,
            'b_acc': balanced_accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    }
    return metrics_dict


def save_results(results: pd.DataFrame, file_name: str) -> None:
    file_name = datetime.now().strftime(f"{file_name}   %d-%m-%Y   %H-%M-%S")
    folder_path = 'metrics'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    results.to_csv(f'{folder_path}/{file_name}.csv')


''' Regression metrics '''
def mape(real: pd.Series, predicted: pd.Series) -> np.float64:
    return np.float64(mean_absolute_percentage_error(real, predicted))

def mse(real: pd.Series, predicted: pd.Series) -> np.float64:
    return np.float64(mean_squared_error(real, predicted, squared=True))

def rmse(real: pd.Series, predicted: pd.Series) -> np.float64:
    return np.float64(mean_squared_error(real, predicted, squared=False))

def r2(real: pd.Series, predicted: pd.Series) -> np.float64:
    return np.float64(r2_score(real, predicted))


''' Classification metrics '''
def accuracy(real: pd.Series, predicted: pd.Series) -> np.float64:
    return np.float64(accuracy_score(real, predicted))

def balanced_accuracy(real: pd.Series, predicted: pd.Series) -> np.float64:
    return np.float64(balanced_accuracy_score(real, predicted))

def f1(real: pd.Series, predicted: pd.Series) -> np.float64:
    return np.float64(f1_score(real, predicted))

def precision(real: pd.Series, predicted: pd.Series) -> np.float64:
    return np.float64(precision_score(real, predicted))

def recall(real: pd.Series, predicted: pd.Series) -> np.float64:
    return np.float64(recall_score(real, predicted, average='macro'))
