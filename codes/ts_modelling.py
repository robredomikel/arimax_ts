"""
This script launches the tsa forecasting phase
"""

import os
import pandas as pd
import numpy as np

from commons import DATA_PATH


def arimax_model(df_path, project_name, periodicity):
    """
    Performs the modelling of the ARIMAX model.

    :param df_path: Path of the existing csv file with project data
    :param project_name: Name of the project
    :param periodicity: Periodicity level between observations
    :return model assessment metrics
    """
    


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
        arimax_results_path = os.path.join(DATA_PATH, "results", "arimax")
        if not os.path.exists(arimax_results_path):
            os.mkdir(arimax_results_path)
            os.mkdir(os.path.join(arimax_results_path, "monthly_results"))
            os.mkdir(os.path.join(arimax_results_path, "biweekly_results"))

        monthly_results_path = os.path.join(arimax_results_path, "monthly_results", project)
        biweekly_results_path = os.path.join(arimax_results_path, "biweekly_results", project)

        if os.path.exists(monthly_results_path) and os.path.exists(biweekly_results_path):
            continue

        # biweekly_statistics = arimax_model()
        # Need to at results in the correct format to the DF

        monthly_statistics = arimax_model()