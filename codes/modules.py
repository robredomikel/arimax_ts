import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
import os
import chardet
from io import StringIO
import pandas as pd


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


def assessmentMetrics(model, predicted_vals, testing_vals, pro_name):
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

    return format_results([model, mape_val, mse_val, mae_val, rmse_val])



