"""
This script launches the tsa forecasting phase
"""

import os
import pandas as pd
import numpy as np

from commons import DATA_PATH


def ts_models():
    """
    Executes the tsa process
    """

    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data")
    output_path = os.path.join(DATA_PATH, "results")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # List existing data files:
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)

    assessment_statistics = ["PROJECT", "MAPE", "MSE", "MAE", "RMSE", "AIC", "BIC"]
    biweekly_assessment = pd.DataFrame(columns=assessment_statistics)
    monthly_assessment = pd.DataFrame(columns=assessment_statistics)

    for i in range(len(biweekly_files)):

        project = biweekly_files[i][:-4]
        monthly_results_path

