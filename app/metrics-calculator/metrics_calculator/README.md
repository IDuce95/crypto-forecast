## Instrukcja korzystania z kalkulatora metryk
***

#### 1. Należy zaimportować główną funkcję poprzez komendę:
`from metrics_calculator.main import calculate_metrics`
***

#### 2. Dane wejściowe, dla których mają być policzone metryki, należy podać w formie słownika:
```
data = {
    'train': {
        'y_real': pd.Series(),
        'y_pred': pd.Series()
    },
    'val': {
        'y_real': pd.Series(),
        'y_pred': pd.Series()
    },
    'test': {
        'y_real': pd.Series(),
        'y_pred': pd.Series()
    }
}
```
#### Kalkulator przyjmuje jedynie dane jako obiekty pd.Series. Podanie obiektów innego typu skutkuje błędem.
#### Funkcja przyjmuje jedynie klucze 'train', 'val' i 'test'. Jeśli znajdzie się inny klucz, to zostanie wyrzucony błąd. Mogą też pojawić jedynie klucze 'train' i 'val' lub dowolna inna kombinacja wymienionych trzech.
***

#### 3. Drugim słownikiem, wchodzącym do funkcji jest słownik z metadanymi:
```
metadata = {
    'problem_type': str,
    'metrics': Tuple[str, ...],
    'results_file_name': str,
    'results_structure': str,
    'comment': str
    'save': bool
}
```
***

#### 4. Należy określić typ problemu: regresja lub klasyfikacja. Typ problemu określamy poprzez:
`problem_type: 'regression'`
#### lub:
`problem_type: 'classification'`
#### Jakakolwiek inna wartość będzie skutkowała błędem.
***

#### 5. Trzeba określić, które metryki mają zostać sprawdzone. W tym celu należy zdefiniować zmienną, która zawiera krotkę z nazwami metryk:
`metrics: ('rmse', 'r2')`

#### Poniżej przedstawiono zaimplementowane metryki oraz ich nazwy, które należy zamieścić w polu `metrics`:
- dla problemu regresji:
    - MAPE: używamy stringa 'mape'
    - MSE: 'mnse'
    - RMSE: 'rmse'
    - R2: 'r2'
- dla problemu klasyfikacji:
    - accuracy: 'acc'
    - balanced accuracy: 'b_acc'
    - F1 score: 'f1'
    - precision: 'precision
    - recall: 'recall'
***

#### 6. Należy również podać nazwę, pod którą zostanie zapisany plik csv, zawierający wyniki. Zostanie utworzony folder o nazwie `metrics`, w którym utworzony zostanie plik. W nazwie pliku (oprócz podanej nazwy) znajdzie się również aktualna data i godzina. Przykładowo:
`file_name: 'metrics_regression'`
***

#### 7. Następnie należy określić strukturę wyników - w jakiej postaci chcemy je otrzymać. Możliwości to:
`results_structure: 'single_row'`
#### co skutkuje utworzeniem wyników w postaci:

| index | comment   | nazwa_metryki_1 - nazwa_zbioru_1 | nazwa_metryki_1 - nazwa_zbioru_2 | ... | nazwa_metryki_n - nazwa_zbioru_3 |
|-------|-----------|----------------------------------|----------------------------------|-----|----------------------------------|
| 0     | komentarz | wartosc                          | wartosc                          | ... | wartosc                          |

#### Na przykład:
| index | comment   | rmse - train | r2 - train | rmse - val | r2 - val | rmse - test | r2 - test |
|-------|-----------|--------------|------------|------------|----------|-------------|-----------|
| 0     | komentarz | 20      | 0.8    | 22    | 0.78  | 25     | 0.75   |

#### lub:

`results_structure: 'multiple_rows'`

#### wtedy wyniki wyglądają tak: 
| index          | metryka_1 | ... | metryka_n |
|----------------|-----------|-----|-----------|
| nazwa_zbioru_1 | wartosc   | ... | wartosc   |
| nazwa_zbioru_2 | wartosc   | ... | wartosc   |
| nazwa_zbioru_3 | wartosc   | ... | wartosc   |

#### Na przykład:
| index | acc     | recall  | precision |
|-------|---------|---------|-----------|
| train | 0.95 | 0.93 | 0.9   |
| val   | 0.9 | 0.9 | 0.85   |
| test  | 0.8 | 0.89 | 0.8   |

#### Pierwsza struktura pozwala na łatwiejsze połączenie wyników, kiedy odpalamy kilka modeli i dla każdego tworzymy osobny plik csv z wynikami. Wtedy w `comment` możemy wpisać informację, która identyfikuje dany model. 
#### Druga struktura jest bardziej przejrzysta, kiedy sprawdzamy tylko jeden model i chcemy poznać dla niego wartości metryk.
***


#### 8. Należy jeszcze podać wartość pola `comment`, które odpowiada za treść, która pojawi się w wynikach, natomiast jedynie gdy wybrana struktura wyników to `single_row`. W przypadku struktury `multiple_rows` komentarz się nie pojawia. Na przykład:
`comment: 'przykładowy komentarz'`
***
#### 9. Trzeba na koniec określić czy chcemy zapisać wyniki do pliku csv. Robimy to poprzez określenie wartości pola `save` jako `True` lub `False`.

***
#### 10. W ostatnim kroku trzeba wywołać funkcję, podając jako argumenty kolejno: słownik z danymi, typ problemu, nazwę pliku oraz krotkę z metrykami. Przykład:
`results = calculate_metrics(data=data, metadata=metadata)`
#### Otrzymany obiekt `results` będzie obiektem typu pandas DataFrame
####

***
***
***

## Walidacja
#### Do modułu została napisana również walidacja, która sprawdza:
- czy wybrana struktura danych to `'single_row'` lub `'multiple_rows'`,
- czy klucze podane w słowniku `data` to 'train', 'val' lub 'test',
- czy podane obiekty są typu pd.Series
- czy podane wartości rzeczywiste i predykcje są typem float
- czy indeksy w szeregach predykcji i wartości rzeczywistych się zgadzają
- czy podany typ problemu to `regression` lub `classification`; w przeciwnym wypadku wyrzuca błąd,
- czy zadane metryki są dostępne dla zadanego typu problemu; wyrzuci błąd na przykład w momencie liczenia metryki `accuracy` dla regresji,
- czy któryś z podanych szeregów nie jest pusty (jego długość to 0),
- czy długość szeregu z wartościami rzeczywistymi jest równa długości szeregu z predykcjami (w przeciwnym wypadku niemożliwe będzie policzenie metryk),
- czy któryś z podanych szeregów nie zawiera brakujących wartości (nan),
- czy słownik metadata zawiera wszystkie wymagane klucze.
####
***
***
***
## Testy
#### Dodatkowo utworzone zostały testy, które sprawdzają poprawność obliczania metryk oraz poprawność wyrzucania błędu każdego typu. Testy znajdują się w pliku `metric_calculator_test.py` w folderze `unit_tests`.
***
***
***
## Logger
#### Dodatkowo w kalkulatorze został również zaimplementowany logger. W folderze `metrics_calculator` znajdzie się folder o nazwie `logs`, w którym znajdować się będą pliki z logami. Każdy taki plik w nazwie będzie miał datę i godzinę jego utworzenia. Jeśli podczas korzystania z kalkulatora metryk jakikolwiek z testów nie przejdzie i tym samym wyrzuci błąd, to zostanie w pliku zapisany odpowiedni komunikat na temat tego błędu. 

#### Oprócz tego błędy wyrzucane w konsoli są dzięki temu odpowiednio opisane i podają informację, który test się nie udał, w której linijce wystąpił problem, ogólny opis błędu oraz opis, czego konkretnie błąd dotyczył.