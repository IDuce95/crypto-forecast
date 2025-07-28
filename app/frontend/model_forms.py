

import streamlit as st
from typing import Dict, Any, Tuple
from .frontend_config import MODEL_DEFAULTS, OPTIMIZATION_DEFAULTS

def _render_optimization_params() -> Tuple[int, int]:

    st.write("**Optimization parameters:**")
    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        init_points = st.number_input(
            "Initial points",
            min_value=OPTIMIZATION_DEFAULTS['init_points_min'],
            max_value=OPTIMIZATION_DEFAULTS['init_points_max'],
            value=OPTIMIZATION_DEFAULTS['init_points'],
            help="Number of initial random points for Bayesian optimization"
        )
    with opt_col2:
        n_iter = st.number_input(
            "Iterations",
            min_value=OPTIMIZATION_DEFAULTS['n_iter_min'],
            max_value=OPTIMIZATION_DEFAULTS['n_iter_max'],
            value=OPTIMIZATION_DEFAULTS['n_iter'],
            help="Number of optimization iterations"
        )

    return init_points, n_iter

def render_decision_tree_form() -> Tuple[Dict[str, Any], bool]:

    st.write("**Configure parameter ranges for optimization:**")

    defaults = MODEL_DEFAULTS['decision_tree']

    col1, col2 = st.columns(2)
    with col1:
        dt_max_depth_min = st.number_input("Max depth - Min", min_value=1, max_value=50, value=defaults['max_depth_min'])
        dt_min_samples_split_min = st.number_input("Min samples split - Min", min_value=2, max_value=50, value=defaults['min_samples_split_min'])
        dt_min_samples_leaf_min = st.number_input("Min samples leaf - Min", min_value=1, max_value=50, value=defaults['min_samples_leaf_min'])

    with col2:
        dt_max_depth_max = st.number_input("Max depth - Max", min_value=1, max_value=50, value=defaults['max_depth_max'])
        dt_min_samples_split_max = st.number_input("Min samples split - Max", min_value=2, max_value=50, value=defaults['min_samples_split_max'])
        dt_min_samples_leaf_max = st.number_input("Min samples leaf - Max", min_value=1, max_value=50, value=defaults['min_samples_leaf_max'])

    init_points, n_iter = _render_optimization_params()
    button_clicked = st.button("Optimize Decision Tree")

    params = {
        "dt_max_depth_min": dt_max_depth_min,
        "dt_max_depth_max": dt_max_depth_max,
        "dt_min_samples_split_min": dt_min_samples_split_min,
        "dt_min_samples_split_max": dt_min_samples_split_max,
        "dt_min_samples_leaf_min": dt_min_samples_leaf_min,
        "dt_min_samples_leaf_max": dt_min_samples_leaf_max,
        "init_points": init_points,
        "n_iter": n_iter
    }

    return params, button_clicked

def render_random_forest_form() -> Tuple[Dict[str, Any], bool]:

    st.write("**Configure parameter ranges for optimization:**")

    defaults = MODEL_DEFAULTS['random_forest']

    col1, col2 = st.columns(2)
    with col1:
        rf_n_estimators_min = st.number_input("N estimators - Min", min_value=1, max_value=500, value=defaults['n_estimators_min'])
        rf_max_depth_min = st.number_input("Max depth - Min", min_value=1, max_value=50, value=defaults['max_depth_min'])
        rf_min_samples_split_min = st.number_input("Min samples split - Min", min_value=2, max_value=50, value=defaults['min_samples_split_min'])

    with col2:
        rf_n_estimators_max = st.number_input("N estimators - Max", min_value=1, max_value=500, value=defaults['n_estimators_max'])
        rf_max_depth_max = st.number_input("Max depth - Max", min_value=1, max_value=50, value=defaults['max_depth_max'])
        rf_min_samples_split_max = st.number_input("Min samples split - Max", min_value=2, max_value=50, value=defaults['min_samples_split_max'])

    init_points, n_iter = _render_optimization_params()
    button_clicked = st.button("Optimize Random Forest")

    params = {
        "rf_n_estimators_min": rf_n_estimators_min,
        "rf_n_estimators_max": rf_n_estimators_max,
        "rf_max_depth_min": rf_max_depth_min,
        "rf_max_depth_max": rf_max_depth_max,
        "rf_min_samples_split_min": rf_min_samples_split_min,
        "rf_min_samples_split_max": rf_min_samples_split_max,
        "init_points": init_points,
        "n_iter": n_iter
    }

    return params, button_clicked

def render_xgboost_form() -> Tuple[Dict[str, Any], bool]:

    st.write("**Configure parameter ranges for optimization:**")

    defaults = MODEL_DEFAULTS['xgboost']

    col1, col2 = st.columns(2)
    with col1:
        xgb_n_estimators_min = st.number_input("N estimators - Min", min_value=1, max_value=500, value=defaults['n_estimators_min'])
        xgb_max_depth_min = st.number_input("Max depth - Min", min_value=1, max_value=20, value=defaults['max_depth_min'])
        xgb_learning_rate_min = st.number_input("Learning rate - Min", min_value=0.001, max_value=1.0, value=defaults['learning_rate_min'], step=0.001)

    with col2:
        xgb_n_estimators_max = st.number_input("N estimators - Max", min_value=1, max_value=500, value=defaults['n_estimators_max'])
        xgb_max_depth_max = st.number_input("Max depth - Max", min_value=1, max_value=20, value=defaults['max_depth_max'])
        xgb_learning_rate_max = st.number_input("Learning rate - Max", min_value=0.001, max_value=1.0, value=defaults['learning_rate_max'], step=0.001)

    init_points, n_iter = _render_optimization_params()
    button_clicked = st.button("Optimize XGBoost")

    params = {
        "xgb_n_estimators_min": xgb_n_estimators_min,
        "xgb_n_estimators_max": xgb_n_estimators_max,
        "xgb_max_depth_min": xgb_max_depth_min,
        "xgb_max_depth_max": xgb_max_depth_max,
        "xgb_learning_rate_min": xgb_learning_rate_min,
        "xgb_learning_rate_max": xgb_learning_rate_max,
        "init_points": init_points,
        "n_iter": n_iter
    }

    return params, button_clicked

def render_lasso_form() -> Tuple[Dict[str, Any], bool]:

    st.write("**Configure parameter ranges for optimization:**")

    defaults = MODEL_DEFAULTS['lasso']

    col1, col2 = st.columns(2)
    with col1:
        lasso_alpha_min = st.number_input("Alpha - Min", min_value=0.001, max_value=100.0, value=defaults['alpha_min'], step=0.001, format="%.3f")
        lasso_max_iter_min = st.number_input("Max iter - Min", min_value=100, max_value=10000, value=defaults['max_iter_min'])
        lasso_tol_min = st.number_input("Tolerance - Min", min_value=0.0001, max_value=1.0, value=defaults['tol_min'], step=0.0001, format="%.4f")

    with col2:
        lasso_alpha_max = st.number_input("Alpha - Max", min_value=0.001, max_value=100.0, value=defaults['alpha_max'], step=0.001, format="%.3f")
        lasso_max_iter_max = st.number_input("Max iter - Max", min_value=100, max_value=10000, value=defaults['max_iter_max'])
        lasso_tol_max = st.number_input("Tolerance - Max", min_value=0.0001, max_value=1.0, value=defaults['tol_max'], step=0.0001, format="%.4f")

    init_points, n_iter = _render_optimization_params()
    button_clicked = st.button("Optimize Lasso")

    params = {
        "lasso_alpha_min": lasso_alpha_min,
        "lasso_alpha_max": lasso_alpha_max,
        "lasso_max_iter_min": lasso_max_iter_min,
        "lasso_max_iter_max": lasso_max_iter_max,
        "lasso_tol_min": lasso_tol_min,
        "lasso_tol_max": lasso_tol_max,
        "init_points": init_points,
        "n_iter": n_iter
    }

    return params, button_clicked

def render_model_parameter_form(model_type: str) -> Tuple[Dict[str, Any], bool]:

    if model_type == "Decision Tree":
        return render_decision_tree_form()
    elif model_type == "Random Forest":
        return render_random_forest_form()
    elif model_type == "XGBoost":
        return render_xgboost_form()
    elif model_type == "Lasso":
        return render_lasso_form()
    else:
        return {}, False
