"""
Combines the needed results into aggregated values for global comparison
"""

from commons import DATA_PATH, assessment_statistics, final_table_columns, FINAL_TS_TABLE_COLS
from modules import transform_to_latex

import pandas as pd
import os
from statistics import mean


def check_results(periodicity_level):
    """
    Checks the format of the results obtain for each model in the provided periodicity level.
    NOTE: Only works for TS approaches
    :param periodicity_level: Given periodicity to examine
    """

    result_directories = ['sarimax_results', 'arimax_results', 'sarima_lm_results',
                          'arima_lm_results', 'results/resuls_ML_sarimax']

    for result_directory in result_directories:

        # Check detected errors in output files
        project_list = os.listdir(os.path.join(DATA_PATH, result_directory))
        for i in range(len(project_list)):
            project_name = project_list[i][:-4]
            project_path = os.path.join(DATA_PATH, result_directory, project_list[i])
            df = pd.read_csv(project_path)
            results_cols = df.columns.tolist()

            # Check if the results file has the correct column names (due to unknown reasons some files got the names
            # inside the results)
            if results_cols != FINAL_TS_TABLE_COLS:
                df.columns = FINAL_TS_TABLE_COLS

            # Check length of the df, should be one project per csv
            if len(df) != 1:
                model_name_col = df['PROJECT']
                model_idx = model_name_col.get_loc(project_name)
                output_row = df.loc[model_idx, :]
                # We overwrite the existing output
                if not os.path.exists(os.path.join(DATA_PATH, f"{result_directory}_clean")):
                    os.mkdir(os.path.join(DATA_PATH, f"{result_directory}_clean"))
                output_row.to_csv(os.path.join(DATA_PATH, f"{result_directory}_clean", project_list[i]),
                                  mode='w', columns=FINAL_TS_TABLE_COLS, index=False)
            else:
                df.to_csv(os.path.join(DATA_PATH, f"{result_directory}_clean", project_list[i]), index=False)

            print("> Project [{}] cleaned for model results [{}] - {}/{}".format(project_name,
                                                                               result_directory,
                                                                               i+1, len(project_list)))


def merge_ml(periodicity):
    """
    Merges the results from the completed
    """

    final_format_dict = {col: [] for col in final_table_columns}

    ml_results_path = os.path.join(DATA_PATH, 'results', 'results_ML_sarimax')
    models_list = os.listdir(ml_results_path)

    # Parse and collect all the assessment files of each of the ML models with the results of each project. Get the mean
    for i in range(len(models_list)):
        model = models_list[i]
        model_path = os.path.join(ml_results_path, model, periodicity)
        assessment_df = pd.read_csv(os.path.join(model_path, 'assessment.csv'))
        final_format_dict['Approach'].append(model)
        if periodicity == 'complete':
            if model is 'svr' or model is 'xgb' or model is 'rf':
                final_format_dict['Type'].append('NL')
            else:
                final_format_dict['Type'].append('L')
        else:
            final_format_dict['Type'].append('~TD')
        final_format_dict['MAPE (%)'].append(round(assessment_df['MAPE'].mean(), 2))
        final_format_dict['MAE'].append(round(assessment_df['MAE'].mean(), 2))
        final_format_dict['MSE'].append(round(assessment_df['MSE'].mean(), 2))
        final_format_dict['RMSE'].append(round(assessment_df['RMSE'].mean(), 2))

        print("> Model [{}] results merged - {}/{}".format(model, i+1, len(models_list)))

    if periodicity == 'complete':
        # Convert into a df
        return pd.DataFrame.from_dict(final_format_dict)
    else:
        return final_format_dict


def save_results(latex, df, file_name):
    """
    Stores the results into csv and in case in latex format
    """

    # 1. Store into csv format
    df.to_csv(os.path.join(DATA_PATH, 'final_results', file_name), index=False)
    # 2. Store into latex format
    if latex:
        transform_to_latex(os.path.join(DATA_PATH, file_name))


def merge_all(periodicity_level):
    """
    Merges the biweekly and monthly results from all the implemented models in the concerning project
    """

    result_directories = ['sarimax_results_clean', 'arimax_results_clean', 'sarima_lm_results_clean',
                          'arima_lm_results_clean', 'results/resuls_ML_sarimax']
    final_results_dict = {col: [] for col in final_table_columns}

    for j in range(len(result_directories)):
        result_directory = result_directories[j]
        if result_directory is 'results/resuls_ML_sarimax':

            # For the given periodicity, check for the approach the obtained results pero project
            prov_df_dict = merge_ml(periodicity=periodicity_level)

            # Store the avg values in the final results dict.
            final_results_dict['Approach'].append(result_directory)
            final_results_dict['Type'].append('~TD')
            final_results_dict['MAPE (%)'].append(round(mean(prov_df_dict['MAPE']), 2))
            final_results_dict['MAE'].append(round(mean(prov_df_dict['MAE']), 2))
            final_results_dict['MSE'].append(round(mean(prov_df_dict['MSE']), 2))
            final_results_dict['RMSE'].append(round(mean(prov_df_dict['RMSE']), 2))

        else:

            # For the given periodicity, check for the approach the obtained results pero project
            prov_df_dict = {col: [] for col in assessment_statistics}
            project_results_dir_path = os.path.join(DATA_PATH, result_directory, f"{periodicity_level}_results")
            project_results_files = os.listdir(project_results_dir_path)
            for i in range(len(project_results_files)):
                project_results_file = project_results_files[i]
                file_df = pd.read_csv(os.path.join(project_results_dir_path, project_results_file))
                prov_df_dict['PROJECT'].append(file_df['PROJECT'][0])
                prov_df_dict['MAPE'].append(file_df['MAPE'][1])
                prov_df_dict['MAE'].append(file_df['MAE'][2])
                prov_df_dict['MSE'].append(file_df['MSE'][3])
                prov_df_dict['RMSE'].append(file_df['RMSE'][4])

                print(f"> PROJECT: {project_results_file} results added to common path - {i+1}/{len(project_results_files)}")

            # Store the avg values in the final results dict.
            if result_directory is "sarimax_results" or result_directory is "sarima_lm_results":
                final_results_dict['Type'].append('STD')
            else:
                final_results_dict['Type'].append('TD')
            final_results_dict['Approach'].append(result_directory)
            final_results_dict['MAPE (%)'].append(round(mean(prov_df_dict['MAPE']), 2))
            final_results_dict['MAE'].append(round(mean(prov_df_dict['MAE']), 2))
            final_results_dict['MSE'].append(round(mean(prov_df_dict['MSE']), 2))
            final_results_dict['RMSE'].append(round(mean(prov_df_dict['RMSE']), 2))

        print(f"> Final results file generated for approach {result_directory} - {j+1}/{len(result_directories)}")

    return pd.DataFrame.from_dict(final_results_dict)


def merge_results(periodicity_level):
    """
    Merges the results of the used models into the aggregated values for global comparison anchored to the according
    periodicity level
    """

    if periodicity_level == "complete":  # Meaning that the data has only been processed by ML models.

        # We deal with the complete data separately as we only need the averages of the ML models
        results_complete_df = merge_ml(periodicity=periodicity_level)
        save_results(latex=True, df=results_complete_df, file_name="original_data_results.csv")

    else:

        results_df = merge_all(periodicity_level=periodicity_level)
        save_results(latex=True, df=results_df, file_name=f"{periodicity_level}_data_results.csv")


def ts_comparison():
    """

    """


def combine_results():
    """
    Combines the obtained results from each of the models into the same file for comparison purposes
    """

    periodicity_levels = ['biweekly', 'monthly', 'complete']

    if not os.path.exists(os.path.join(DATA_PATH, 'final_results')):
        os.mkdir(os.path.join(DATA_PATH, 'final_results'))

    # merge_outcomes into biweekly data, monthly and complete data
    for periodicity in periodicity_levels:
        check_results(periodicity_level=periodicity)
        merge_results(periodicity_level=periodicity)

    # Visualization stage
    # 1. Comparison among Seasonal Time Series and non-seasonal time series. (bar plots)
    # ts_comparison()

    # 2. Comparison among Seasonal Time Series and non-seasonal ML models (spider charts)



