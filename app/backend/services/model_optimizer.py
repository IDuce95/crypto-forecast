

from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

from app.backend.services.service_registry import registry

class ModelOptimizer:

    def __init__(self):
        self.data_manager = registry.get_data_manager()
        self.metrics_calculator = registry.get_metrics_calculator()

    def optimize_decision_tree(
        self,
        dataset_name: str,
        prediction_horizon: int,
        max_depth_min: int = 2,
        max_depth_max: int = 20,
        min_samples_split_min: int = 2,
        min_samples_split_max: int = 20,
        min_samples_leaf_min: int = 1,
        min_samples_leaf_max: int = 20,
        init_points: int = 5,
        n_iter: int = 5,
        split_ratios: Dict[str, float] = None
    ) -> Dict[str, Any]:

        if split_ratios is None:
            split_ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}

        data_info = self.data_manager.prepare_data_with_visualization_info(
            dataset_name, prediction_horizon, split_ratios
        )
        X_train, X_val, X_test = data_info['X_train'], data_info['X_val'], data_info['X_test']
        y_train, y_val, y_test = data_info['y_train'], data_info['y_val'], data_info['y_test']

        def dt_objective(max_depth, min_samples_split, min_samples_leaf):
            max_depth = int(max_depth) if max_depth > 1 else None
            min_samples_split = int(min_samples_split)
            min_samples_leaf = int(min_samples_leaf)

            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return -mean_absolute_percentage_error(y_val, y_pred)

        pbounds = {
            'max_depth': (max_depth_min, max_depth_max),
            'min_samples_split': (min_samples_split_min, min_samples_split_max),
            'min_samples_leaf': (min_samples_leaf_min, min_samples_leaf_max)
        }

        optimizer = BayesianOptimization(f=dt_objective, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        best_params = optimizer.max['params']
        best_params['max_depth'] = int(best_params['max_depth']) if best_params['max_depth'] > 1 else None
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

        model = DecisionTreeRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        metrics = self.metrics_calculator.calculate_all_metrics(
            y_train, y_val, y_test, train_pred, val_pred, test_pred
        )

        return {
            'best_params': best_params,
            'metrics': metrics,
            'optimization_score': optimizer.max['target'],
            'predictions': {
                'train': train_pred.tolist(),
                'validation': val_pred.tolist(),
                'test': test_pred.tolist()
            },
            'actual_values': {
                'train': y_train.tolist(),
                'validation': y_val.tolist(),
                'test': y_test.tolist()
            },
            'visualization_data': {
                'original_values': data_info['original_values'].tolist(),
                'original_dates': data_info['original_dates'],
                'train_end_idx': data_info['train_end_idx'],
                'val_end_idx': data_info['val_end_idx'],
                'split_ratios': split_ratios
            }
        }

    def optimize_random_forest(
        self,
        dataset_name: str,
        prediction_horizon: int,
        n_estimators_min: int = 10,
        n_estimators_max: int = 200,
        max_depth_min: int = 2,
        max_depth_max: int = 20,
        min_samples_split_min: int = 2,
        min_samples_split_max: int = 20,
        init_points: int = 5,
        n_iter: int = 5,
        split_ratios: Dict[str, float] = None
    ) -> Dict[str, Any]:

        if split_ratios is None:
            split_ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}

        data_info = self.data_manager.prepare_data_with_visualization_info(
            dataset_name, prediction_horizon, split_ratios
        )
        X_train, X_val, X_test = data_info['X_train'], data_info['X_val'], data_info['X_test']
        y_train, y_val, y_test = data_info['y_train'], data_info['y_val'], data_info['y_test']

        def rf_objective(n_estimators, max_depth, min_samples_split):
            n_estimators = int(n_estimators)
            max_depth = int(max_depth) if max_depth > 1 else None
            min_samples_split = int(min_samples_split)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return -mean_absolute_percentage_error(y_val, y_pred)

        pbounds = {
            'n_estimators': (n_estimators_min, n_estimators_max),
            'max_depth': (max_depth_min, max_depth_max),
            'min_samples_split': (min_samples_split_min, min_samples_split_max)
        }

        optimizer = BayesianOptimization(f=rf_objective, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth']) if best_params['max_depth'] > 1 else None
        best_params['min_samples_split'] = int(best_params['min_samples_split'])

        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        metrics = self.metrics_calculator.calculate_all_metrics(
            y_train, y_val, y_test, train_pred, val_pred, test_pred
        )

        return {
            'best_params': best_params,
            'metrics': metrics,
            'optimization_score': optimizer.max['target'],
            'predictions': {
                'train': train_pred.tolist(),
                'validation': val_pred.tolist(),
                'test': test_pred.tolist()
            },
            'actual_values': {
                'train': y_train.tolist(),
                'validation': y_val.tolist(),
                'test': y_test.tolist()
            },
            'visualization_data': {
                'original_values': data_info['original_values'].tolist(),
                'original_dates': data_info['original_dates'],
                'train_end_idx': data_info['train_end_idx'],
                'val_end_idx': data_info['val_end_idx'],
                'split_ratios': split_ratios
            }
        }

    def optimize_xgboost(
        self,
        dataset_name: str,
        prediction_horizon: int,
        n_estimators_min: int = 10,
        n_estimators_max: int = 200,
        max_depth_min: int = 2,
        max_depth_max: int = 10,
        learning_rate_min: float = 0.01,
        learning_rate_max: float = 0.3,
        init_points: int = 5,
        n_iter: int = 5,
        split_ratios: Dict[str, float] = None
    ) -> Dict[str, Any]:

        if split_ratios is None:
            split_ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}

        data_info = self.data_manager.prepare_data_with_visualization_info(
            dataset_name, prediction_horizon, split_ratios
        )
        X_train, X_val, X_test = data_info['X_train'], data_info['X_val'], data_info['X_test']
        y_train, y_val, y_test = data_info['y_train'], data_info['y_val'], data_info['y_test']

        def xgb_objective(n_estimators, max_depth, learning_rate):
            n_estimators = int(n_estimators)
            max_depth = int(max_depth)

            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return -mean_absolute_percentage_error(y_val, y_pred)

        pbounds = {
            'n_estimators': (n_estimators_min, n_estimators_max),
            'max_depth': (max_depth_min, max_depth_max),
            'learning_rate': (learning_rate_min, learning_rate_max)
        }

        optimizer = BayesianOptimization(f=xgb_objective, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])

        model = XGBRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        metrics = self.metrics_calculator.calculate_all_metrics(
            y_train, y_val, y_test, train_pred, val_pred, test_pred
        )

        return {
            'best_params': best_params,
            'metrics': metrics,
            'optimization_score': optimizer.max['target'],
            'predictions': {
                'train': train_pred.tolist(),
                'validation': val_pred.tolist(),
                'test': test_pred.tolist()
            },
            'actual_values': {
                'train': y_train.tolist(),
                'validation': y_val.tolist(),
                'test': y_test.tolist()
            },
            'visualization_data': {
                'original_values': data_info['original_values'].tolist(),
                'original_dates': data_info['original_dates'],
                'train_end_idx': data_info['train_end_idx'],
                'val_end_idx': data_info['val_end_idx'],
                'split_ratios': split_ratios
            }
        }

    def optimize_lasso(
        self,
        dataset_name: str,
        prediction_horizon: int,
        alpha_min: float = 0.001,
        alpha_max: float = 10.0,
        max_iter_min: int = 100,
        max_iter_max: int = 5000,
        tol_min: float = 0.0001,
        tol_max: float = 0.01,
        init_points: int = 5,
        n_iter: int = 5,
        split_ratios: Dict[str, float] = None
    ) -> Dict[str, Any]:

        if split_ratios is None:
            split_ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}

        data_info = self.data_manager.prepare_data_with_visualization_info(
            dataset_name, prediction_horizon, split_ratios
        )
        X_train, X_val, X_test = data_info['X_train'], data_info['X_val'], data_info['X_test']
        y_train, y_val, y_test = data_info['y_train'], data_info['y_val'], data_info['y_test']

        def lasso_objective(alpha, max_iter, tol):
            max_iter = int(max_iter)

            model = Lasso(
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return -mean_absolute_percentage_error(y_val, y_pred)

        pbounds = {
            'alpha': (alpha_min, alpha_max),
            'max_iter': (max_iter_min, max_iter_max),
            'tol': (tol_min, tol_max)
        }

        optimizer = BayesianOptimization(f=lasso_objective, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        best_params = optimizer.max['params']
        best_params['max_iter'] = int(best_params['max_iter'])

        model = Lasso(**best_params, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        metrics = self.metrics_calculator.calculate_all_metrics(
            y_train, y_val, y_test, train_pred, val_pred, test_pred
        )

        return {
            'best_params': best_params,
            'metrics': metrics,
            'optimization_score': optimizer.max['target'],
            'predictions': {
                'train': train_pred.tolist(),
                'validation': val_pred.tolist(),
                'test': test_pred.tolist()
            },
            'actual_values': {
                'train': y_train.tolist(),
                'validation': y_val.tolist(),
                'test': y_test.tolist()
            },
            'visualization_data': {
                'original_values': data_info['original_values'].tolist(),
                'original_dates': data_info['original_dates'],
                'train_end_idx': data_info['train_end_idx'],
                'val_end_idx': data_info['val_end_idx'],
                'split_ratios': split_ratios
            }
        }
