import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
import os
import chardet
from io import StringIO
import pandas as pd
import statsmodels.api as sm
from commons import DATA_PATH
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def RSS(y, X, model):
    """
    Theoretical calculation of RSS
    :param y: dependent variable vector
    :param X: Independent variables matrix
    :param model: regression model included
    """
    return np.sum((y - model.predict(X))**2)


def AIC(n, k, rss):
    """
    Theoretical calculation of AIC
    :param n: number of matrix rows
    :param k: number of model parameters (intercept included)
    :param rss: RSS value from the fitted model
    """
    return n * np.log(rss/n) + 2 * k


def detect_existing_output(project, paths, flag_num, files_num, approach):

    monthly_results_path = paths[0]
    biweekly_results_path = paths[1]
    if len(paths) == 3:  # Means the call comes from the ML modelling stage.
        complete_results_path = paths[2]
        # Check if the project has already been processed
        if (os.path.exists(monthly_results_path) and os.path.exists(biweekly_results_path) and
                os.path.exists(complete_results_path)):
            print(f"> PROJECT: {project} has already been processed - NEXT {flag_num+1}/{files_num}")
            return True
        else:
            print(f"> Processing project {project} for {approach} approach.")
            return False

    else:
        # Check if the project has already been processed
        if os.path.exists(monthly_results_path) and os.path.exists(biweekly_results_path):
            print(f"> PROJECT: {project} has already been processed - NEXT {flag_num+1}/{files_num}")
            return True
        else:
            if os.path.exists(monthly_results_path):
                print(f"> Only monthy analysis processed for PROJECT: {project}")
            elif os.path.exists(biweekly_results_path):
                print(f"> Only biweekly analysis processed for PROJECT: {project}")
            else:
                print(f"> Processing project {project} for {approach} approach.")

            return False


def format_results(result_list):
    """
    Makes the results fit the required format in pandas. Goes from list to dict
    """

    new_row = pd.DataFrame([{"PROJECT": result_list[0], "MAPE": result_list[1], "MSE": result_list[2],
                             "MAE": result_list[3], "RMSE": result_list[4]}])
    return new_row


def change_encoding(path):
    """
    Changes the encoding style of the df provided
    """

    with open(path, 'rb') as f:
        content = f.read()  # Read the file in binary mode
    f.close()
    decoded_content = content.decode('windows-1252', errors='ignore')
    complete_df = pd.read_csv(StringIO(decoded_content))
    return complete_df


def check_encoding(path):
    """
    Check the encoding style of the df provided:
    """

    with open(path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    f.close()

    return encoding


def MAPE(predicted_vals, testing_vals):
    y_true, y_pred = np.array(testing_vals), np.array(predicted_vals)
    return np.mean(np.abs((y_true - y_pred)/y_true)) * 100


def MAE(predicted_vals, testing_vals):
    y_true, y_pred = np.array(testing_vals), np.array(predicted_vals)
    return np.mean(np.abs((y_true - y_pred)))


def MSE(predicted_vals, testing_vals):
    y_true, y_pred = np.array(testing_vals), np.array(predicted_vals)
    return mean_squared_error(y_true, y_pred)


def RMSE(predicted_vals, testing_vals):
    y_true, y_pred = np.array(testing_vals), np.array(predicted_vals)
    return np.sqrt(MSE(y_true, y_pred))


def assessmentMetrics(predicted_vals, testing_vals, pro_name):
    """

    Calculates MAPE, MSE, MAE, RMSE & LogLoss
    :param model:
    :param predicted_vals:
    :param testing_vals:
    :param pro_name:
    :return:
    """

    mape_val = MAPE(predicted_vals=predicted_vals, testing_vals=testing_vals)
    mse_val = mean_squared_error(testing_vals, predicted_vals)
    mae_val = mean_absolute_error(testing_vals, predicted_vals)
    rmse_val = RMSE(testing_vals=testing_vals, predicted_vals=predicted_vals)

    return format_results([pro_name, mape_val, mse_val, mae_val, rmse_val])


def transform_to_latex(df_path):
    """
    Transforms the pandas dataframe to latex
    """

    df = pd.read_csv(df_path)
    latex_df = df.to_latex(index=False)
    results_type = df_path.split('/')[-1][:-4]
    file_tex = results_type + '.tex'
    file_path = os.path.join(DATA_PATH, 'final_results', file_tex)
    with open(file_path, 'w') as f:
        f.write(latex_df)
    f.close()
    print(f"{results_type} RESULTS table saved into LaTex format!")


def decomposition_plot(proname_clean, biweekly_data, monthly_data, decomposition_path):

    # Monthly
    decomposition = sm.tsa.seasonal_decompose(monthly_data['SQALE_INDEX'],
                                              model='additive',
                                              period=12)
    decomposition.plot()
    plt.savefig(os.path.join(decomposition_path, "monthly_plots", f"{proname_clean}.pdf"))
    plt.close()

    # Biweekly
    decomposition = sm.tsa.seasonal_decompose(biweekly_data['SQALE_INDEX'],
                                              model='additive',
                                              period=26)
    decomposition.plot()
    plt.savefig(os.path.join(decomposition_path, "biweekly_plots", f"{proname_clean}.pdf"))
    plt.close()


def create_diagnostics(seasonality, periodicity, best_model_params, best_regressors, training_df, project_name):
    """
    Generates model diagnostics for the given best model in each case
    """

    X_train = training_df[best_regressors].astype(float)
    X_train_scaled = X_train.map(np.log1p)
    y_train = training_df['SQALE_INDEX'].astype(float)
    arima_order = best_model_params[0]
    s_order = best_model_params[1]

    if seasonality:
        model = SARIMAX(y_train.to_numpy(), exog=X_train_scaled.to_numpy(), order=arima_order,
                        seasonal_order=s_order, enforce_stationarity=True, enforce_invertibility=True)

    else:
        model = SARIMAX(y_train.to_numpy(), exog=X_train_scaled.to_numpy(), order=arima_order,
                        enforce_stationarity=True, enforce_invertibility=True)

    fitted_model = model.fit(disp=0)

    # Generate paths
    diagnostics_path = os.path.join(DATA_PATH, 'model_diagnostic_plots')
    sarimax_path = os.path.join(diagnostics_path, "sarimax")
    arimax_path = os.path.join(diagnostics_path, "arimax")

    # Perform seasonal decomposition plots for all projects
    if not os.path.exists(diagnostics_path):
        os.mkdir(diagnostics_path)
        os.mkdir(os.path.join(diagnostics_path, "sarimax"))
        os.mkdir(os.path.join(diagnostics_path, "arimax"))
        os.mkdir(os.path.join(diagnostics_path, "sarimax", "biweekly"))
        os.mkdir(os.path.join(diagnostics_path, "arimax", "biweekly"))
        os.mkdir(os.path.join(diagnostics_path, "sarimax", "monthly"))
        os.mkdir(os.path.join(diagnostics_path, "arimax", "monthly"))

    # Visualization
    plt.rcParams['axes.labelsize'] = 17
    plt.rcParams['axes.titlesize'] = 17
    plt.rcParams['xtick.labelsize'] = 17
    plt.rcParams['ytick.labelsize'] = 17
    plt.rcParams['legend.fontsize'] = 15

    if seasonality == True:

        fig = fitted_model.plot_diagnostics(figsize=(10, 10))  # Initial: 8,7
        plt.savefig(os.path.join(sarimax_path, periodicity, f'{project_name}.pdf'), format='pdf')
        plt.close(fig)

    else:

        fig = fitted_model.plot_diagnostics(figsize=(10, 10))  # Initial: 8,7
        plt.savefig(os.path.join(arimax_path, periodicity, f'{project_name}.pdf'), format='pdf')
        plt.close(fig)
