"""
This project executes a series of TSA techniques as well as ML models

NOTE: The data collection and preprocessing has been already performed.
"""

import os
from ml_modelling import ml_models
from ts_modelling import ts_models
from commons import DATA_PATH


def main():

    results_path = os.path.join(DATA_PATH, 'results')
    if not os.path.exists(results_path):
        os.mkdir(results_path)