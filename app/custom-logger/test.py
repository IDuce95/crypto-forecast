""" Imports required for the logger to work """
import inspect
import os
import sys
from datetime import datetime

import logger_utils.exceptions as exceptions
import logger_utils.logger as logger_module

""" Create a directory with log files """
PATH_TO_LOGS = "./logs"
if not os.path.exists(PATH_TO_LOGS):
    os.makedirs(PATH_TO_LOGS)

""" Logger initialization """
log_path = datetime.now().strftime(f"{PATH_TO_LOGS}/%d-%m-%Y   %H-%M-%S.log")
logger = logger_module.get_module_logger(
    mod_name=__name__,
    file_name=os.path.basename(__file__),
    log_path=log_path,
    level=1
)


class Test:
    """ Example use of logger """

    def __init__(self):
        try:
            self.model = "model"
        except Exception as err:
            logger_module.log_error(
                logger,
                exceptions.InitializationError,
                err,
                inspect.stack()[0].function,
                sys.exc_info()[-1].tb_lineno,
            )

    def preprocessing(self):
        try:
            X = 10
            return X
        except Exception as err:
            logger_module.log_error(
                logger,
                exceptions.PreprocessingError,
                err,
                inspect.stack()[0].function,
                sys.exc_info()[-1].tb_lineno,
            )

    def predict(self, X):
        try:
            y = X
            return y
        except Exception as err:
            logger_module.log_error(
                logger,
                exceptions.PredictionError,
                err,
                inspect.stack()[0].function,
                sys.exc_info()[-1].tb_lineno,
            )

    def postprocessing(self, y):
        try:
            return y
        except Exception as err:
            logger_module.log_error(
                logger,
                exceptions.PostprocessingError,
                err,
                inspect.stack()[0].function,
                sys.exc_info()[-1].tb_lineno,
            )

    def run_pipeline(self):
        X = self.preprocessing()
        y = self.predict(X)
        y = self.postprocessing(y)

        return y


def testowanie():
    dataset = "example"
    horizon = 10
    metric = 0.1
    model_type = "Random Forest"
    hiperparams = {"max_depth": 5, "learning_rate": 0.01}

    logger_module.log_result(logger, dataset, horizon, metric,
                             model_type, hiperparams)

    logger_module.log_info(logger, "Info level only contains the message")

    logger_module.log_debug(
        logger,
        "Debug and warning levels contain file name and function name",
        inspect.stack()[0].function,
    )

    logger_module.log_warning(
        logger,
        "Error level contains error message and code line",
        inspect.stack()[0].function,
    )

    try:
        print(non_existant_variable)
    except Exception as err:
        logger_module.log_error(
            logger,
            exceptions.DoesNotExist,
            err,
            inspect.stack()[0].function,
            sys.exc_info()[-1].tb_lineno,
        )


Test().run_pipeline()
testowanie()
