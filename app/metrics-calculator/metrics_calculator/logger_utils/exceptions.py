from metrics_calculator.logger_utils.colors import Colors


class Error(Exception):
    pass


def get_full_message(func_name, code_line, description, message):
    if func_name == '<module>':
        func_name = 'Błąd poza funkcją'
    else:
        func_name = f'{func_name}()'

    full_message = f"\n\n{Colors().WARNING}===>" \
+ f"{Colors().ENDCOLOR} Błąd w funkcji: \
{Colors().ERROR}{func_name}{Colors().ENDCOLOR}" \
+ f"\n{Colors().WARNING}===>{Colors().ENDCOLOR} Błąd w linijce: \
{Colors().ERROR}{code_line}{Colors().ENDCOLOR}" \
+ f"\n{Colors().WARNING}===>{Colors().ENDCOLOR} Opis błędu: \
{Colors().ERROR}'{description}'{Colors().ENDCOLOR}" \
+ f"\n{Colors().WARNING}===>{Colors().ENDCOLOR} Treść błędu: \
{Colors().ERROR}'{message}'{Colors().ENDCOLOR}"
    return full_message


class ProblemTypeNotImplemented(Error):
    def __init__(self, func_name, code_line, message,
                 description='Taki typ problemu nie \
został zaimplementowany') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class MetricNotImplemented(Error):
    def __init__(self, func_name, code_line, message,
                 description='Taka metryka nie została zaimplementowana \
dla tego typu problemu') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class EmptyArray(Error):
    def __init__(self, func_name, code_line, message,
                 description='Wykryto pustą tablicę') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class IncompatibleLengths(Error):
    def __init__(self, func_name, code_line, message,
                 description='Niezgodność długości tablic') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class NanValuesDetected(Error):
    def __init__(self, func_name, code_line, message,
                 description='Wykryto wartości nan') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class WrongType(Error):
    def __init__(self, func_name, code_line, message,
                 description='Niepoprawny typ danych wejściowych') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class WrongValuesType(Error):
    def __init__(self, func_name, code_line, message,
                 description='Niepoprawny typ wartości \
wewnątrz obiektów pd.Series') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class IndexesNotMatch(Error):
    def __init__(self, func_name, code_line, message,
                 description='Niezgodność indeksów w obiektach \
predykcji rzeczywistych i predykcji') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class WrongDatasetName(Error):
    def __init__(self, func_name, code_line, message,
                 description='Niepoprawna nazwa klucza w słowniku') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class WrongResultsStructureName(Error):
    def __init__(self, func_name, code_line, message,
                 description='Niepoprawna nazwa struktury wyników \
w słowniku') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)


class MetadataKeyMissing(Error):
    def __init__(self, func_name, code_line, message,
                 description='Brakujący klucz w słowniku metadata') -> None:
        self.message = get_full_message(func_name, code_line,
                                        description, message)
        super().__init__(self.message)