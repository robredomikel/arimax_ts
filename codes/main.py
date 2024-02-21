"""
This project executes a series of TSA techniques as well as ML models

NOTE: The data collection and preprocessing has been already performed.
"""

import os
from ml_modelling import ml_models
from ts_modelling import ts_models
from related_work import related_models
from commons import DATA_PATH, SARIMAX, RELATED_WORK, ML_MODELS


def main():

    # SARIMAX modelling stage execustion
    if SARIMAX:
        ts_models()

    # SARIMA + LM related work stage execution
    if RELATED_WORK:
        related_models()

    # ML stage
    if ML_MODELS:
        ml_models()