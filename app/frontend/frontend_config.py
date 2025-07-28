

PLOT_COLORS = {
    'actual_data': "
    'predictions': "
    'split_lines': '
}

PLOT_STYLE = {
    'actual_data': {
        'linewidth': 2,
        'alpha': 1.0,
    },
    'predictions': {
        'linewidth': 2,
        'alpha': 0.4,
        'linestyle': '-',
    },
    'split_lines': {
        'linewidth': 2,
        'alpha': 0.7,
    },
    'figure_size': (16, 8),
    'title_fontsize': 18,
    'label_fontsize': 16,
    'info_fontsize': 12,
}

MODEL_DEFAULTS = {
    'decision_tree': {
        'max_depth_min': 2,
        'max_depth_max': 20,
        'min_samples_split_min': 2,
        'min_samples_split_max': 20,
        'min_samples_leaf_min': 1,
        'min_samples_leaf_max': 20,
    },
    'random_forest': {
        'n_estimators_min': 10,
        'n_estimators_max': 200,
        'max_depth_min': 2,
        'max_depth_max': 20,
        'min_samples_split_min': 2,
        'min_samples_split_max': 20,
    },
    'xgboost': {
        'n_estimators_min': 10,
        'n_estimators_max': 200,
        'max_depth_min': 2,
        'max_depth_max': 10,
        'learning_rate_min': 0.01,
        'learning_rate_max': 0.3,
    },
    'lasso': {
        'alpha_min': 0.001,
        'alpha_max': 10.0,
        'max_iter_min': 100,
        'max_iter_max': 5000,
        'tol_min': 0.0001,
        'tol_max': 0.01,
    }
}

OPTIMIZATION_DEFAULTS = {
    'init_points': 5,
    'n_iter': 5,
    'init_points_min': 1,
    'init_points_max': 20,
    'n_iter_min': 1,
    'n_iter_max': 50,
}

DATA_SPLIT_DEFAULTS = {
    'train': 60,
    'validation': 20,
    'test': 20,
    'train_min': 10,
    'train_max': 80,
    'val_min': 5,
    'val_max': 30,
    'test_min': 5,
    'test_max': 30,
    'step': 5,
}

PREDICTION_DEFAULTS = {
    'horizon_default': 1,
    'horizon_min': 1,
    'horizon_max': 30,
}

LAYOUT_CONFIG = {
    'columns_ratio': [0.25, 0.25, 0.5],
    'gap': "large",
    'border': True,
}

MESSAGES = {
    'split_error': "⚠️ Total split ({total}%) exceeds 100%! Please adjust the values.",
    'split_warning': "ℹ️ Total split is {total}%. Remaining {remaining}% will be unused.",
    'split_success': "✅ Perfect split: {total}%",
    'no_viz_data': "No visualization data available for this result."
}

PATHS = {
    'style_file': 'plt_style.mlpstyle',
}

AVAILABLE_MODELS = ["Choose model...", "Decision Tree", "Random Forest", "XGBoost", "Lasso"]
