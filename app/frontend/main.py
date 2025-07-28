import streamlit as st
import sys
import os
sys.path.append(os.getcwd())

from app import tools
from app.config import get_config
from app.logger import logger
from app.frontend.api_client import APIClient
from app.frontend.ui_components import (
    render_layout_sections,
    render_dataset_selection,
    render_model_training_section,
    render_model_parameter_form,
    display_optimization_results
)

st.set_page_config(layout="wide")
config = get_config()
api_client = APIClient(config)

logger.info(f"Starting Streamlit frontend in {config.current_env} environment")

@st.cache_data
def get_cached_datasets() -> list:
    return tools.get_available_datasets(config)

dataset_management, model_training, results = render_layout_sections()

dataset, split_ratios = render_dataset_selection(dataset_management, get_cached_datasets)

prediction_horizon, model_type = render_model_training_section(model_training)

if model_type != "Choose model...":
    with model_training:
        params, button_clicked = render_model_parameter_form(model_type)

        if button_clicked and params:
            body = {
                "dataset_name": dataset,
                "prediction_horizon": prediction_horizon,
                "split_ratios": split_ratios,
                **params
            }

            endpoint_map = {
                "Decision Tree": config.api_endpoints.optimize_dt_endpoint,
                "Random Forest": config.api_endpoints.optimize_rf_endpoint,
                "XGBoost": config.api_endpoints.optimize_xgb_endpoint,
                "Lasso": config.api_endpoints.optimize_lasso_endpoint
            }

            endpoint = endpoint_map.get(model_type)
            if endpoint:
                result = api_client.make_optimization_request(
                    model_type, endpoint, body, dataset, prediction_horizon
                )

                if result:
                    with results:
                        display_optimization_results(result, dataset)
