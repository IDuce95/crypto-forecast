import inspect
import os
import sys
from datetime import datetime
from typing import Tuple

import pandas as pd

import metrics_calculator.config as config
import metrics_calculator.logger_utils.exceptions as exceptions
import metrics_calculator.logger_utils.logger as logger_module


""" Utworzenie katalogu z logami """
if not os.path.exists(config.PATH_TO_LOGS):
    os.makedirs(config.PATH_TO_LOGS)


""" Wczytanie loggera """
log_file_name = f"{config.PATH_TO_LOGS}/%d-%m-%Y   %H-%M-%S.log"
log_path = datetime.now().strftime(log_file_name)
logger = logger_module.get_module_logger(mod_name=__name__,
                                         file_name=os.path.basename(__file__),
                                         log_path=log_path,
                                         level=1)


def run_validation(
    real: pd.Series,
    predicted: pd.Series,
    metrics: Tuple[str, ...],
    metrics_dict: dict,
    problem_type: str,
    dataset: str,
    results_structure: str
) -> None:

    validate_results_structure(results_structure)
    validate_dataset_name(dataset)
    validate_problem_type(problem_type)
    validate_metrics(metrics, problem_type, metrics_dict)
    validate_series(real, predicted, dataset)


def validate_results_structure(results_structure: str) -> None:
    message = f'Struktura wyników "{results_structure}" jest nieprawidłowa. \
Poprawne nazwy struktury to: "single_row" oraz "multiple_rows"'
    try:
        assert results_structure in ['single_row', 'multiple_rows']
    except AssertionError:
        logger_module.log_error(logger, exceptions.WrongResultsStructureName,
                                message,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_dataset_name(dataset: str) -> None:
    message = f'Klucz "{dataset}" w słowniku jest nieprawidłowy. \
Poprawne nazwy kluczy to: "train", "val" oraz "test".'
    try:
        assert dataset in ['train', 'val', 'test']
    except AssertionError:
        logger_module.log_error(logger, exceptions.WrongDatasetName,
                                message,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_series(real: pd.Series,
                    predicted: pd.Series,
                    dataset: str) -> None:

    validate_type(real, predicted, dataset)
    validate_values_types(real, predicted, dataset)
    validate_not_empty(real, predicted, dataset)
    validate_compatible_lengths(real, predicted, dataset)
    validate_index_match(real, predicted, dataset)
    validate_nan_values(real, predicted, dataset)


def validate_type(real: pd.Series,
                  predicted: pd.Series,
                  dataset: str) -> None:

    message_real = f'Obiekt wartości rzeczywistych nie jest obiektem \
typu pd.Series w zbiorze "{dataset}"'
    message_pred = f'Obiekt predykcji nie jest obiektem typu pd.Series \
w zbiorze "{dataset}"'

    try:
        assert isinstance(real, pd.Series)
    except AssertionError:
        logger_module.log_error(logger, exceptions.WrongType,
                                message_real,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore

    try:
        assert isinstance(predicted, pd.Series)
    except AssertionError:
        logger_module.log_error(logger, exceptions.WrongType,
                                message_pred,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_values_types(real: pd.Series,
                          predicted: pd.Series,
                          dataset: str) -> None:

    message_real = f'Obiekt wartości rzeczywistych dla zbioru "{dataset}" \
zawiera wartości o niepoprawnym typie. Wymagany typ danych to float'
    message_pred = f'Obiekt predykcji dla zbioru "{dataset}" \
zawiera wartości o niepoprawnym typie. Wymagany typ danych to float'

    try:
        assert (real.map(type) == float).all()
    except AssertionError:
        logger_module.log_error(logger, exceptions.WrongValuesType,
                                message_real,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore

    try:
        assert (predicted.map(type) == float).all()
    except AssertionError:
        logger_module.log_error(logger, exceptions.WrongValuesType,
                                message_pred,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_index_match(real: pd.Series,
                         predicted: pd.Series,
                         dataset: str) -> None:
    message = f'Dla zbioru {dataset} indeksy obiektów pd.Series nie są zgodne'

    try:
        assert (real.index == predicted.index).all()
    except AssertionError:
        logger_module.log_error(logger, exceptions.IndexesNotMatch,
                                message,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_not_empty(real: pd.Series,
                       predicted: pd.Series,
                       dataset: str) -> None:
    message_real = f'Długość tablicy z wartościami rzeczywistymi \
jest zerowa dla zbioru "{dataset}"'
    message_pred = f'Długość tablicy z predykcjami \
jest zerowa dla zbioru "{dataset}"'

    try:
        assert len(real) != 0
    except AssertionError:
        logger_module.log_error(logger, exceptions.EmptyArray,
                                message_real,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore

    try:
        assert len(predicted) != 0
    except AssertionError:
        logger_module.log_error(logger, exceptions.EmptyArray,
                                message_pred,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_compatible_lengths(real: pd.Series,
                                predicted: pd.Series,
                                dataset: str) -> None:
    message = f'''Ilość wartości rzeczywistych ({len(real)}) \
nie zgadza się z ilością predykcji ({len(predicted)}) dla zbioru "{dataset}"'''
    try:
        assert len(real) == len(predicted)
    except AssertionError:
        logger_module.log_error(logger, exceptions.IncompatibleLengths,
                                message,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_nan_values(real: pd.Series,
                        predicted: pd.Series,
                        dataset: str) -> None:
    message_real = f'Tablica wartości rzeczywistych zawiera \
wartości nan dla zbioru "{dataset}"'
    message_pred = f'Tablica predykcji zawiera \
wartości nan dla zbioru "{dataset}"'

    try:
        assert real.isna().sum() == 0
    except AssertionError:
        logger_module.log_error(logger, exceptions.NanValuesDetected,
                                message_real,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore

    try:
        assert predicted.isna().sum() == 0
    except AssertionError:
        logger_module.log_error(logger, exceptions.NanValuesDetected,
                                message_pred,
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_metrics(metrics: Tuple[str, ...],
                     problem_type: str,
                     metrics_dict: dict) -> None:

    for metric in metrics:
        try:
            assert metric in metrics_dict[problem_type].keys()
        except AssertionError:
            logger_module.log_error(logger, exceptions.MetricNotImplemented,
                                    f'Błąd dla metryki "{metric}"',
                                    inspect.stack()[0].function,
                                    sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_problem_type(problem_type: str) -> None:
    try:
        assert problem_type in ['regression', 'classification']
    except AssertionError:
        logger_module.log_error(logger, exceptions.ProblemTypeNotImplemented,
                                f'Błąd dla typu problemu "{problem_type}"',
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore


def validate_metadata_dict(metadata: dict) -> None:

    keys = ['problem_type', 'metrics', 'results_file_name',
            'results_structure', 'comment', 'save']
    try:
        for key in keys:
            assert key in metadata.keys()
    except AssertionError:
        logger_module.log_error(logger, exceptions.MetadataKeyMissing,
                                f'W słowniku metadata brakuje klucza "{key}"',
                                inspect.stack()[0].function,
                                sys.exc_info()[-1].tb_lineno)  # type: ignore
