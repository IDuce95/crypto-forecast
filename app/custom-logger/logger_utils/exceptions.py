from typing import NoReturn

from logger_utils.colors import Colors


class Error(Exception):
    pass


class InitializationError(Error):
    def __init__(self, func_name, code_line, message,
                 description='Initialization error') -> NoReturn:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class PreprocessingError(Error):
    def __init__(self, func_name, code_line, message,
                 description='Preprocessing error') -> NoReturn:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class PredictionError(Error):
    def __init__(self, func_name, code_line, message,
                 description='Prediction error') -> NoReturn:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class PostprocessingError(Error):
    def __init__(self, func_name, code_line, message,
                 description='Postprocessing error') -> NoReturn:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class DoesNotExist(Error):
    def __init__(self, func_name, code_line, message,
                 description='Variable does not exist') -> NoReturn:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


def get_full_message(func_name, code_line, description, message):
    if func_name == '<module>':
        func_name = 'Error outside the function'
    else:
        func_name = f'{func_name}()'

    full_message = f"\n\n{Colors().WARNING}===>" \
+ f"{Colors().ENDCOLOR} Error in function: \
{Colors().ERROR}{func_name}{Colors().ENDCOLOR}" \
+ f"\n{Colors().WARNING}===>{Colors().ENDCOLOR} Error in line: \
{Colors().ERROR}{code_line}{Colors().ENDCOLOR}" \
+ f"\n{Colors().WARNING}===>{Colors().ENDCOLOR} Error description: \
{Colors().ERROR}'{description}'{Colors().ENDCOLOR}" \
+ f"\n{Colors().WARNING}===>{Colors().ENDCOLOR} Error message: \
{Colors().ERROR}'{message}'{Colors().ENDCOLOR}"
    return full_message
