"""
This script launches the related work forecasting phase
"""

import os
import pandas as pd
import numpy as np
import pmdarima as pmd

from commons import DATA_PATH


def relwork_model(df_path, project_name, periodicity):
    """
    Performs the modelling of the ARIMA + LM model.

    :param df_path: Path of the existing csv file with project data
    :param project_name: Name of the project
    :param periodicity: Periodicity level between observations
    :return model assessment metrics
    """

    print(f"> Processing project {project_name}")
    # Read the dataframe
    df = pd.read_csv(df_path)
    df.COMMIT_DATE = pd.to_datetime(df.COMMIT_DATE)

    # Dependent variable
    sqale_index = df.SQALE_INDEX.to_numpy()

    # Independent variables
    xregressors = df.iloc[:, 2:].to_numpy()

    # Initial data splitting.
    split_point = round(len(sqale_index))
    training_sqale = sqale_index[:split_point]
    testing_sqale = sqale_index[split_point:]
    training_xregressors = xregressors[:split_point]
    testing_xregressors = xregressors[split_point:]




def related_models():
    """
    Executes the tsa process with the related work models
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
        relwork_results_path = os.path.join(DATA_PATH, "results", "related_work")
        if not os.path.exists(relwork_results_path):
            os.mkdir(relwork_results_path)
            os.mkdir(os.path.join(relwork_results_path, "monthly_results"))
            os.mkdir(os.path.join(relwork_results_path, "biweekly_results"))

        monthly_results_path = os.path.join(relwork_results_path, "monthly_results", project)
        biweekly_results_path = os.path.join(relwork_results_path, "biweekly_results", project)

        if os.path.exists(monthly_results_path) and os.path.exists(biweekly_results_path):
            continue

        biweekly_statistics = relwork_model(df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
                                           project_name=project,
                                           periodicity="biweekly")
        # Need to at results in the correct format to the DF

        monthly_statistics = relwork_model(df_path=os.path.join(monthly_results_path, monthly_files[i]),
                                          project_name=project,
                                          periodicity="monthly")
