import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from metrics_calculator.app import calculate_metrics



results = calculate_metrics(data=data, metadata=metadata)
print('[x] running for regression...')
print(results)



results = calculate_metrics(data=data, metadata=metadata)
print('[x] running for classification...')
print(results)


def generate_data(problem_type):
    generating_function_dict = {
        'classification': make_classification(n_samples=100, n_features=3),
        'regression': make_regression(n_samples=100, n_features=3, coef=True)
    }

    X_train, y_train = generating_function_dict[problem_type]
    X_val, y_val = generating_function_dict[problem_type]
    X_test, y_test = generating_function_dict[problem_type]

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_model(problem_type, X_train, y_train):
    models_dict = {
        'classification': DecisionTreeClassifier(max_depth=3),
        'regression': DecisionTreeRegressor(max_depth=3)
    }

    model = models_dict[problem_type]
    model.fit(X_train, y_train)
    return model


def get_data_structure(model, X_train, y_train, X_val, y_val, X_test, y_test):

    data = {
        'train': {
            'y_real': pd.Series(y_train, dtype=float),
            'y_pred': pd.Series(model.predict(X_train), dtype=float)
        },
        'val': {
            'y_real': pd.Series(y_val, dtype=float),
            'y_pred': pd.Series(model.predict(X_val), dtype=float)
        },
        'test': {
            'y_real': pd.Series(y_test, dtype=float),
            'y_pred': pd.Series(model.predict(X_test), dtype=float)
        }
    }

    return data


def get_metadata(problem_type):
    metadata = {
        'problem_type': 'regression',
        'metrics': ('rmse', 'r2', 'mape', 'mse'),
        'results_file_name': 'metrics_regression',
        'results_structure': 'single_row',
        'comment': 'Komentarz testowy',
        'save': False
    }

    metadata = {
        'problem_type': 'classification',
        'metrics': ('acc', 'b_acc', 'recall', 'f1', 'precision'),
        'results_file_name': 'metrics_classification',
        'results_structure': 'multiple_rows',
        'comment': 'Komentarz testowy',
        'save': False
    }

    metrics_dict = {
        'classification'
    }


def run_single_problem_type(problem_type):
    X_train, y_train, X_val, y_val, X_test, y_test = generate_data(problem_type)
    model = get_model(problem_type, X_train, y_train)
    data = get_data_structure(model, X_train, y_train, X_val, y_val, X_test, y_test)
    metadata = get_metadata(problem_type)


def main():
    for problem_type in ['classification', 'regression']:
        run_single_problem_type(problem_type)


if __name__ == '__main__':
    main()
