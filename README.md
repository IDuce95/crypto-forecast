# Crypto Forecasting

Machine learning system for cryptocurrency price prediction using technical indicators and time series analysis.

## Overview
This project provides automated cryptocurrency price forecasting using multiple ML algorithms with comprehensive data preprocessing, feature engineering, and experiment tracking.

## Features
- Advanced data preprocessing with technical indicators
- Multiple ML models (Random Forest, XGBoost, Decision Tree, Lasso)
- MLflow experiment tracking and model management
- REST API for model training and predictions
- Streamlit web interface
- Docker containerization

## Quick Start
```bash
make api          # Start REST API (port 5000)
make streamlit    # Start web interface (port 8501)
```

## Architecture
- **Backend**: FastAPI REST API
- **Frontend**: Streamlit web interface  
- **ML Pipeline**: Scikit-learn, XGBoost models
- **Tracking**: MLflow experiment management
- **Database**: SQLite for data storage
