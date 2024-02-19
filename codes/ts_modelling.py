"""
This script launches the tsa forecasting phase
"""

import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json

from commons import DATA_PATH


def backward_modelling(df, periodicity):
    """

    """

    # Define the ranges for d and D since we are manually iterating over these
    d_range = D_range = range(0, 3)
    if periodicity == "monthly":
        s = 12  # Seasonal period
    else:
        s = 26  # Bi-weekly periodicity

    best_aic = np.inf
    best_model_cfg = None
    best_regressors = None

    # Iterate over d and D values
    for d in d_range:
        for D in D_range:
            # Use auto_arima to find the best p, q, P, Q given d and D
            try:
                auto_arima_model = auto_arima(df['SQALE_INDEX'], start_p=1, start_q=1,
                                              max_p=5, max_q=5, d=d, D=D, start_P=1, start_Q=1,
                                              max_P=2, max_Q=2, m=s, seasonal=True,
                                              stepwise=True, suppress_warnings=True,
                                              error_action='ignore', trace=False)

                # Extract the best ARIMA order and seasonal order found by auto_arima
                p, q = auto_arima_model.order[0], auto_arima_model.order[2]
                P, Q = auto_arima_model.seasonal_order[0], auto_arima_model.seasonal_order[2]

                print(f"Best p, q combination: {p} {q} - Seasonal: {P} {Q}")
                print(f"d: {d}, D: {D}")
                # Begin backward selection of regressors
                current_regressors = df.iloc[:, 2:].columns.tolist()
                while current_regressors:
                    tmp_X = df[current_regressors]
                    model = SARIMAX(df['SQALE_INDEX'], exog=tmp_X, order=(p, d, q),
                                    seasonal_order=(P, D, Q, s),
                                    enforce_stationarity=True, enforce_invertibility=True)
                    results = model.fit(disp=0)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_model_cfg = ((p, d, q), (P, D, Q, s))
                        best_regressors = current_regressors.copy()

                    if len(current_regressors) > 1:
                        aic_with_regressor_removed = []
                        for regressor in current_regressors:
                            try_regressors = current_regressors.copy()
                            try_regressors.remove(regressor)
                            tmp_X_try = df[try_regressors]
                            model_try = SARIMAX(df['SQALE_INDEX'], exog=tmp_X_try, order=(p, d, q),
                                                seasonal_order=(P, D, Q, s),
                                                enforce_stationarity=True, enforce_invertibility=True)
                            results_try = model_try.fit(disp=0)
                            aic_with_regressor_removed.append((results_try.aic, regressor))
                        aic_with_regressor_removed.sort()
                        current_regressors.remove(aic_with_regressor_removed[0][1])
                    else:
                        break  # Stop if only one regressor left
                    print(f"Number of remaining predictors: {len(current_regressors)}")
            except Exception as e:
                print(f"Error with configuration: {(d, D)} - {str(e)}")
                continue

    print(f"Best SARIMAX{best_model_cfg} - AIC:{best_aic} with regressors {best_regressors}")
    return best_model_cfg, best_aic, best_regressors


def model_testing(training_df, testing_df, best_model_cfg, best_regressors):
    """

    """
    


def arimax_model(df_path, project_name, periodicity):
    """
    Performs the modelling of the ARIMAX model.

    :param df_path: Path of the existing csv file with project data
    :param project_name: Name of the project
    :param periodicity: Periodicity level between observations
    :return model assessment metrics
    """

    df = pd.read_csv(df_path)
    df.COMMIT_DATE = pd.to_datetime(df.COMMIT_DATE)

    # Dependent variable
    sqale_index = df.SQALE_INDEX.to_numpy()

    # Independent variables
    xregressors = df.iloc[:, 2:].to_numpy()

    # Initial data splitting. (80% training 20% testing)
    split_point = round(len(sqale_index)*0.8)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]

    # SARIMAX backward modelling
    best_model_params, best_aic, best_regressors = backward_modelling(df=training_df, periodicity=periodicity)

    # Store the obtained results in json:
    best_model_path = os.path.join(DATA_PATH, "best_models")
    if not os.path.exists(best_model_path):
        os.mkdir(best_model_path)
        os.mkdir(best_model_path, "biweekly")
        os.mkdir(best_model_path, "monthly")

    json_dict = {'model_params': best_model_params, 'best_aic': best_aic, "best_regressors": best_regressors}
    json_object = json.dumps(json_dict, indent=4)
    with open(os.path.join(best_model_path, periodicity, f"{project_name}.json")) as out:
        out.write(json_object)

    # Model testing
    model_testing()



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

        biweekly_statistics = arimax_model(df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
                                           project_name=project,
                                           periodicity="biweekly")

        # Need to at results in the correct format to the DF
        monthly_statistics = arimax_model(df_path=os.path.join(monthly_results_path, monthly_files[i]),
                                          project_name=project,
                                          periodicity="monthly")