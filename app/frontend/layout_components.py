

import streamlit as st
from typing import Dict, Any, Tuple

from .frontend_config import (
    DATA_SPLIT_DEFAULTS, PREDICTION_DEFAULTS, 
    LAYOUT_CONFIG, MESSAGES, AVAILABLE_MODELS
)

def render_layout_sections() -> Tuple[Any, Any, Any]:

    st.title("Crypto forecasting")
    st.header("Service for training models through a REST API.")

    dataset_management, model_training, results = st.columns(
        LAYOUT_CONFIG['columns_ratio'], 
        gap=LAYOUT_CONFIG['gap'], 
        border=LAYOUT_CONFIG['border']
    )

    return dataset_management, model_training, results

def render_dataset_selection(dataset_management_column, get_cached_datasets_func) -> Tuple[str, Dict[str, float]]:

    defaults = DATA_SPLIT_DEFAULTS
    
    with dataset_management_column:
        st.subheader("Dataset management")
        dataset = st.selectbox("Select dataset", get_cached_datasets_func())

        st.write("---")
        st.write("**Data split configuration:**")

        train_split = st.slider(
            "Train split (%)", 
            min_value=defaults['train_min'], 
            max_value=defaults['train_max'], 
            value=defaults['train'], 
            step=defaults['step']
        )
        val_split = st.slider(
            "Validation split (%)", 
            min_value=defaults['val_min'], 
            max_value=defaults['val_max'], 
            value=defaults['validation'], 
            step=defaults['step']
        )
        test_split = st.slider(
            "Test split (%)", 
            min_value=defaults['test_min'], 
            max_value=defaults['test_max'], 
            value=defaults['test'], 
            step=defaults['step']
        )

        total_split = train_split + val_split + test_split

        if total_split > 100:
            st.error(MESSAGES['split_error'].format(total=total_split))
            split_ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
        elif total_split < 100:
            remaining = 100 - total_split
            st.warning(MESSAGES['split_warning'].format(total=total_split, remaining=remaining))
            split_ratios = {
                "train": train_split / 100,
                "validation": val_split / 100,
                "test": test_split / 100
            }
        else:
            st.success(MESSAGES['split_success'].format(total=total_split))
            split_ratios = {
                "train": train_split / 100,
                "validation": val_split / 100,
                "test": test_split / 100
            }

        st.write(f"**Current split:** Train: {train_split}%, Val: {val_split}%, Test: {test_split}%")

    return dataset, split_ratios

def render_model_training_section(model_training_column) -> Tuple[int, str]:

    defaults = PREDICTION_DEFAULTS
    
    with model_training_column:
        st.subheader("Model training")

        prediction_horizon = st.number_input(
            "Prediction horizon (days ahead)",
            min_value=defaults['horizon_min'],
            max_value=defaults['horizon_max'],
            value=defaults['horizon_default'],
            help="Number of days ahead to predict"
        )

        model_type = st.selectbox("Select model", AVAILABLE_MODELS)

    return prediction_horizon, model_type
