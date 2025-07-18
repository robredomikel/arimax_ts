"""
This script launches the tsa forecasting phase
"""

import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json

from commons import DATA_PATH
from modules import MAPE, RMSE, MAE, MSE, check_encoding, detect_existing_output, create_diagnostics


def backward_modelling(df, periodicity, seasonality, output_flag=True):
    """
    Finds the best modelling order for the SARIMAX model and stores its' parameters, AIC value and useful regressors in
    a JSON file
    """

    # Dependent variable
    sqale_index = df.SQALE_INDEX.to_numpy()
    # Initial data splitting.
    split_point = round(len(sqale_index)*0.8)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]

    # Define the ranges for d and D since we are manually iterating over these
    """
    if seasonality:
        d_range = D_range = range(0, 3)
    else:
        d_range = range(0, 3)
        D_range = [0]  # We don't look into seasonal component
    """
    if periodicity == "monthly":
        s = 12  # Seasonal period
    else:
        s = 26  # Bi-weekly periodicity

    best_aic = np.inf
    best_model_cfg = None
    best_regressors = None

    # We do not iterate through the considered differentiation parameters as there is not enough computational power
    # to performm this computation.
    try:

        """
        if seasonality:
            auto_arima_model = auto_arima(training_df['SQALE_INDEX'], m=s, seasonal=seasonality,
                                          stepwise=True, suppress_warnings=True,
                                          error_action='ignore', trace=False,
                                          information_criterion='aic')
            P, D, Q = (auto_arima_model.seasonal_order[0], auto_arima_model.seasonal_order[1],
                       auto_arima_model.seasonal_order[2])
        else:
            auto_arima_model = auto_arima(training_df['SQALE_INDEX'], seasonal=seasonality,
                                          stepwise=True, suppress_warnings=True,
                                          error_action='ignore', trace=False,
                                          information_criterion='aic')
            P, D, Q = np.nan
        
        
        # Extract the best ARIMA order and seasonal order found by auto_arima
        p, d, q = auto_arima_model.order[0], auto_arima_model.order[1], auto_arima_model.order[2]

        print(f"Best p, q combination: {p} {q} - Seasonal: {P} {Q}")
        print(f"d: {d}, D: {D}")
        """

        # Begin backward selection of regressors
        current_regressors = training_df.iloc[:, 2:].columns.tolist()
        while current_regressors:
            # tmp_X = training_df[current_regressors]
            # tmp_X_scaled = np.log1p(tmp_X)
            print(f"> REMAINING REGRESSORS: {len(current_regressors)}")

            """
            if seasonality:
                model = SARIMAX(training_df['SQALE_INDEX'], exog=tmp_X_scaled, order=(p, d, q),
                                seasonal_order=(P, D, Q, s),
                                enforce_stationarity=True, enforce_invertibility=True)
            else:
                model = SARIMAX(training_df['SQALE_INDEX'], exog=tmp_X_scaled, order=(p, d, q),
                                enforce_stationarity=True, enforce_invertibility=True)
            """

            print("> Fitting model...")
            if len(current_regressors) > 1:
                aic_with_regressor_removed = []
                i = 0
                for regressor in current_regressors:
                    try_regressors = current_regressors.copy()
                    try_regressors.remove(regressor)
                    tmp_X_try = training_df[try_regressors].to_numpy()
                    tmp_X_try_scaled = np.log1p(tmp_X_try)  # log scaling for highly skewed data

                    try:
                        if seasonality:

                            # First get the best model combinations through the stepwise auto_arima
                            auto_arima_model = auto_arima(training_df['SQALE_INDEX'].to_numpy(), X=tmp_X_try_scaled, m=s, seasonal=seasonality,
                                                          stepwise=True, suppress_warnings=True,
                                                          error_action='ignore', trace=True,
                                                          information_criterion='aic', test='adf')
                            P, D, Q = (auto_arima_model.seasonal_order[0], auto_arima_model.seasonal_order[1],
                                       auto_arima_model.seasonal_order[2])
                            # Extract the best ARIMA order and seasonal order found by auto_arima
                            p, d, q = auto_arima_model.order[0], auto_arima_model.order[1], auto_arima_model.order[2]
                            # Second, obtain SARIMAX model from the obtained parameter combinations.
                            model_try = SARIMAX(training_df['SQALE_INDEX'].to_numpy(), exog=tmp_X_try_scaled, order=(p, d, q),
                                                seasonal_order=(P, D, Q, s),
                                                enforce_stationarity=True, enforce_invertibility=True)
                        else:
                            # First get the best model combinations through the stepwise auto_arima
                            auto_arima_model = auto_arima(training_df['SQALE_INDEX'].to_numpy(), X=tmp_X_try_scaled, m=s, seasonal=seasonality,
                                                          stepwise=True, suppress_warnings=True,
                                                          error_action='ignore', trace=True,
                                                          information_criterion='aic', test='adf')
                            P, D, Q = (auto_arima_model.seasonal_order[0], auto_arima_model.seasonal_order[1],
                                       auto_arima_model.seasonal_order[2])
                            # Extract the best ARIMA order and seasonal order found by auto_arima
                            p, d, q = auto_arima_model.order[0], auto_arima_model.order[1], auto_arima_model.order[2]
                            model_try = SARIMAX(training_df['SQALE_INDEX'].to_numpy(), exog=tmp_X_try_scaled, order=(p, d, q),
                                                enforce_stationarity=True, enforce_invertibility=True)

                        results_try = model_try.fit(disp=0)
                        aic_with_regressor_removed.append((results_try.aic, regressor))

                        # We check which has been so far the best model
                        if results_try.aic < best_aic:
                            best_aic = results_try.aic

                            best_model_cfg = ((p, d, q), (P, D, Q, s))
                            best_regressors = current_regressors.copy()
                    except ConvergenceWarning:
                        print(f"> Failed to converge for model excluding {regressor}. Skipping...")
                        continue

                aic_with_regressor_removed.sort()
                current_regressors.remove(aic_with_regressor_removed[0][1])

                print(f"> ts modelling performed when seasonality: {seasonality} - {i+1}/{len(current_regressors)}")
                i+=1
            else:
                print("> break - all regressor combinations examined")
                break  # Stop if only one regressor left

            print(f"> Number of remaining predictors: {len(current_regressors)}")
    except Exception as e:
        print(f"> Error with configuration: {str(e)}")
        output_flag = False

    if seasonality:
        print(f"> Best SARIMAX{best_model_cfg} - AIC:{best_aic} with regressors {best_regressors}")
    else:
        print(f"> Best ARIMAX{best_model_cfg} - AIC:{best_aic} with regressors {best_regressors}")

    return best_model_cfg, round(best_aic, 2), best_regressors, output_flag


def model_testing(training_df, testing_df, best_model_cfg, best_regressors, seasonality, project):
    """
    Given the best model order parameters obtained from the backward variable selection and aut_arima tuning we
    fit the SARIMAX model for forecasting
    """

    arima_order = best_model_cfg[0]
    s_order = best_model_cfg[1]

    predictions = []

    for i in range(len(testing_df)):
        # Training the SARIMAX model
        X_train = training_df[best_regressors].astype(float)
        y_train = training_df['SQALE_INDEX'].astype(float)
        X_train_scaled = X_train.map(np.log1p)

        # Model fitting
        if seasonality:
            model = SARIMAX(y_train.to_numpy(), exog=X_train_scaled.to_numpy(), order=arima_order,
                            seasonal_order=s_order, enforce_stationarity=True, enforce_invertibility=True)

        else:
            model = SARIMAX(y_train.to_numpy(), exog=X_train_scaled.to_numpy(), order=arima_order,
                            enforce_stationarity=True, enforce_invertibility=True)

        fitted_model = model.fit(disp=0)
        print(f"> model fit {i} times")

        # Model forecasting
        best_reg_df = testing_df[best_regressors]
        best_reg_df_scaled = np.log1p(best_reg_df)
        X_test = best_reg_df_scaled.iloc[i, :].values.reshape(1, -1)
        y_pred = fitted_model.get_forecast(steps=1, exog=X_test)
        predictions.append(y_pred.predicted_mean[0])
        # Expand the training data for next iteration
        new_obs = testing_df.iloc[i, :]
        training_df = pd.concat([training_df, new_obs.to_frame().T], ignore_index=False, axis=0)

    print(f"> Model testing stage performed for project {project}")

    return predictions, round(fitted_model.aic, 2), round(fitted_model.bic, 2)


def assessment_metrics(predictions, real_values):
    """
    Performs the calculations of the statistics defined for performance assessment
    """
    mape_val = round(MAPE(real_values, predictions), 2)
    mse_val = round(MSE(real_values, predictions), 2)
    mae_val = round(MAE(real_values, predictions), 2)
    rmse_val = round(RMSE(real_values, predictions), 2)

    return mape_val, mse_val, mae_val, rmse_val


def arimax_model(df_path, project_name, periodicity, seasonality):
    """
    Performs the modelling of the ARIMAX model.

    :param df_path: Path of the existing csv file with project data
    :param project_name: Name of the project
    :param periodicity: Periodicity level between observations
    :return model assessment metrics
    """

    # DATA PREPARATION (Splitting)
    encoding = check_encoding(df_path)
    df = pd.read_csv(df_path, encoding=encoding)
    df.COMMIT_DATE = pd.to_datetime(df.COMMIT_DATE)
    sqale_index = df.SQALE_INDEX.to_numpy()  # Dependent variable
    split_point = round(len(sqale_index)*0.8)  # Initial data splitting. (80% training 20% testing)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]

    # SARIMAX backward modelling
    best_model_params, best_aic, best_regressors, output_flag = backward_modelling(df=training_df,
                                                                                   periodicity=periodicity,
                                                                                   seasonality=seasonality,
                                                                                   output_flag=True)

    # If there was error in the calculation of the backward modelling then we exclude the results from this project
    # for the given data format.
    if output_flag is False:
        return [project_name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    # Store the obtained results in json:
    if seasonality:
        best_model_path = os.path.join(DATA_PATH, "best_sarimax_models")
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)
            os.mkdir(os.path.join(best_model_path, "biweekly"))
            os.mkdir(os.path.join(best_model_path, "monthly"))
    else:
        best_model_path = os.path.join(DATA_PATH, "best_arimax_models")
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)
            os.mkdir(os.path.join(best_model_path, "biweekly"))
            os.mkdir(os.path.join(best_model_path, "monthly"))

    # Stores the results in a json file
    json_dict = {'model_params': best_model_params, 'best_aic': best_aic, "best_regressors": best_regressors}
    json_object = json.dumps(json_dict, indent=4)
    with open(os.path.join(best_model_path, periodicity, f"{project_name}.json"), 'w+') as out:
        out.write(json_object)

    # Model testing
    try:
        # Creating model diagnostics for visualization:
        create_diagnostics(seasonality, periodicity, best_model_params, best_regressors, training_df, project_name)
        predictions, aic_val, bic_val = model_testing(training_df=training_df, testing_df=testing_df,
                                                      best_model_cfg=best_model_params, best_regressors=best_regressors,
                                                      seasonality=seasonality, project=project_name)
        assessment_vals = assessment_metrics(predictions=predictions, real_values=testing_df["SQALE_INDEX"].tolist())

    except np.linalg.LinAlgError:  # In case there is some lineal algebra decomposition error
        assessment_vals = [np.nan, np.nan, np.nan, np.nan]
        aic_val, bic_val = np.nan, np.nan

    print(f"> Final results for project {project_name}:\n"
          f"MAPE:{assessment_vals[0]}\n"
          f"MSE:{assessment_vals[1]}\n"
          f"MAE:{assessment_vals[2]}\n"
          f"RMSE:{assessment_vals[3]}\n")
    return [project_name, assessment_vals[0], assessment_vals[1], assessment_vals[2], assessment_vals[3],
            aic_val, bic_val]


def ts_models(seasonality):
    """
    Executes the tsa process
    """

    # Check if Seasonality is taken into consideration
    if seasonality == True:
        output_directory = "sarimax_results"
    else:
        output_directory = "arimax_results"

    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data")
    output_path = os.path.join(DATA_PATH, output_directory)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "monthly_results"))
        os.mkdir(os.path.join(output_path, "biweekly_results"))

    # List existing data files:
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)

    assessment_statistics = ["PROJECT", "MAPE", "MSE", "MAE", "RMSE", "AIC", "BIC"]
    for i in range(len(biweekly_files)):

        project = biweekly_files[i][:-4]
        monthly_results_path = os.path.join(output_path, "monthly_results", f"{project}.csv")
        biweekly_results_path = os.path.join(output_path, "biweekly_results", f"{project}.csv")

        biweekly_assessment = pd.DataFrame(columns=assessment_statistics)
        monthly_assessment = pd.DataFrame(columns=assessment_statistics)

        # Check if the project has already been processed
        if detect_existing_output(project=project, paths=[monthly_results_path, biweekly_results_path],
                                  flag_num=i, files_num=len(biweekly_files), approach=f"{seasonality}-ARIMAX"):
            print(f"> Project {project} already procesed for SARIMAX modelling")
            continue

        # Runs the SARIMAX execution for the given project in biweekly format
        print(f"> Processing {project} for biweekly data")
        biweekly_statistics = arimax_model(df_path=os.path.join(biweekly_data_path, biweekly_files[i]),
                                           project_name=project,
                                           periodicity="biweekly",
                                           seasonality=seasonality)

        biweekly_assessment.loc[len(biweekly_assessment)] = biweekly_statistics
        biweekly_assessment.to_csv(biweekly_results_path, index=False)
        print(f"> Processing {project} for monthly data")
        monthly_statistics = arimax_model(df_path=os.path.join(monthly_data_path, monthly_files[i]),
                                          project_name=project,
                                          periodicity="monthly",
                                          seasonality=seasonality)

        monthly_assessment.loc[len(monthly_assessment)] = monthly_statistics
        monthly_assessment.to_csv(monthly_results_path, index=False)

        if seasonality:
            print(f"> SARIMAX modelling for project <{project}> performed - {i+1}/{len(biweekly_files)}")
        else:
            print(f"> ARIMAX modelling for project <{project}> performed - {i+1}/{len(biweekly_files)}")

    if seasonality:
        print("> SARIMAX stage performed!")
    else:
        print("> ARIMAX stage performed!")