

import requests
import streamlit as st
from typing import Dict, Any

from app.logger import logger

class APIClient:

    def __init__(self, config):

        self.config = config

    def build_api_url(self, endpoint: str) -> str:

        return f"http://{self.config.rest_api_settings.backend_hostname}:{self.config.rest_api_settings.port}{endpoint}/"

    def make_optimization_request(
        self,
        model_name: str,
        endpoint: str,
        body: Dict[str, Any],
        dataset: str,
        prediction_horizon: int
    ) -> None:

        logger.info(f"User initiated {model_name} optimization with dataset={dataset}, prediction_horizon={prediction_horizon}")
        api_url = self.build_api_url(endpoint)

        with st.spinner(f'Optimizing {model_name}... This may take a few minutes.'):
            response = requests.post(api_url, json=body)

        if response.status_code == 200:
            result = response.json()
            logger.info(f"{model_name} optimization completed successfully")
            st.success(f"{model_name} optimized and trained on {result['dataset']} with {result['prediction_horizon']}-day horizon!")
            st.info(f"Best parameters found: {result['best_params']}")
            return result
        else:
            logger.warning(f"{model_name} optimization failed: {response.status_code} - {response.text}")
            st.error(f"Error occurred: {response.status_code}: {response.text}")
            return None
