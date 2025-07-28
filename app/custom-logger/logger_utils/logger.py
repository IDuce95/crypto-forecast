import logging
import logging.handlers


"""
Built-in severity levels:
==> DEBUG = 10
==> INFO = 20
==> WARN = 30
==> ERROR = 40
==> CRITICAL = 50
"""


RESULT_LEVEL_NUM = 55
logging.addLevelName(RESULT_LEVEL_NUM, "RESULT")


def result(self, message, *args, **kwargs):
    if self.isEnabledFor(RESULT_LEVEL_NUM):
        self._log(RESULT_LEVEL_NUM, message, args, **kwargs)


logging.Logger.result = result


class MyFormatter(logging.Formatter):
    _formats = {}

    def __init__(self, *args, **kwargs):
        super(MyFormatter, self).__init__(*args, **kwargs)

    def set_formatter(self, level, formatter):
        self._formats[level] = formatter

    def format(self, record):
        if 'code_line' in record.args:
            record.code_line = record.args.get("code_line")
        if 'dataset' in record.args:
            record.dataset = record.args.get("dataset")
        if 'horizon' in record.args:
            record.horizon = record.args.get("horizon")
        if 'func_name' in record.args:
            record.func_name = record.args.get("func_name")
            if record.func_name == '<module>':
                record.func_name = 'none'
            else:
                record.func_name = record.func_name + '()'
        if 'metric' in record.args:
            record.metric = record.args.get("metric")
        if 'model_type' in record.args:
            record.model_type = record.args.get("model_type")
        if 'hiperparams' in record.args:
            record.hiperparams = record.args.get("hiperparams")

        f = self._formats.get(record.levelno)
        if f is None:
            f = super(MyFormatter, self)

        return f.format(record)


def get_module_logger(mod_name: str, file_name: str, log_path: str, level):

    row_for_error = '[%(asctime)s] || \
%(levelname)-7s ||' \
+ f' File name: {file_name} ||' \
+ ' Function name: %(func_name)s || \
Code line: %(code_line)s || \
Message: "%(message)s"'

    row_for_warning = '[%(asctime)s] || \
%(levelname)-7s ||' \
+ f' File name: {file_name} ||' \
+ ' Function name: %(func_name)s || \
Message: "%(message)s"'

    row_for_info = '[%(asctime)s] || \
%(levelname)-7s || \
Message: "%(message)s"'

    row_for_debug = '[%(asctime)s] || \
%(levelname)-7s ||' \
+ f' File name: {file_name} ||' \
+ ' Function name: %(func_name)s || \
Message: "%(message)s"'

    row_for_result = '[%(asctime)s] || \
%(levelname)-7s || \
Dataset: %(dataset)s || \
Horizon: %(horizon)s || \
Metric: %(metric)s || \
Model type: %(model_type)s || \
Hiperparams: %(hiperparams)s'

    formatter = MyFormatter()
    formatter.set_formatter(logging.ERROR, logging.Formatter(row_for_error, "%Y-%m-%d %H:%M:%S"))  # noqa: E501
    formatter.set_formatter(logging.WARNING, logging.Formatter(row_for_warning, "%Y-%m-%d %H:%M:%S"))  # noqa: E501
    formatter.set_formatter(logging.INFO, logging.Formatter(row_for_info, "%Y-%m-%d %H:%M:%S"))  # noqa: E501
    formatter.set_formatter(logging.DEBUG, logging.Formatter(row_for_debug, "%Y-%m-%d %H:%M:%S"))  # noqa: E501
    formatter.set_formatter(RESULT_LEVEL_NUM, logging.Formatter(row_for_result, "%Y-%m-%d %H:%M:%S"))  # noqa: E501

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(mod_name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    return logger


def log_result(logger, dataset, horizon, metric, model_type, hiperparams):
    logger.result(None, {'dataset': dataset,
                         'horizon': horizon,
                         'metric': metric,
                         'model_type': model_type,
                         'hiperparams': hiperparams})


def log_error(logger, ErrorType, err, func_name, code_line):
    logger.error(err, {'code_line': code_line, 'func_name': func_name})
    raise ErrorType(func_name, code_line, err)


def log_warning(logger, message, func_name):
    logger.warning(message, {'func_name': func_name})


def log_info(logger, message):
    logger.info(message)


def log_debug(logger, message, func_name):
    logger.debug(message, {'func_name': func_name})
