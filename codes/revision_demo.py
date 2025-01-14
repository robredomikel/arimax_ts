"""
Performs the predictions with the selected models ARIMAX and SARIMAX with a 70/30 split for survey demo to practitioners
"""

from commons import DATA_PATH
from modules import MAPE, RMSE, MAE, MSE
import os
import pandas as pd
from tqdm import tqdm


def detect_existing_output(project, paths, flag_num, files_num, approach):

    biweekly_results_path = paths
    if len(paths) == 3:  # Means the call comes from the ML modelling stage.
        complete_results_path = paths[2]
        # Check if the project has already been processed
        if os.path.exists(biweekly_results_path) and os.path.exists(complete_results_path):
            print(f"> PROJECT: {project} has already been processed - NEXT {flag_num+1}/{files_num}")
            return True
        else:
            print(f"> Processing project {project} for {approach} approach.")
            return False

    else:
        # Check if the project has already been processed
        if os.path.exists(biweekly_results_path):
            print(f"> PROJECT: {project} has already been processed - NEXT {flag_num+1}/{files_num}")
            return True
        else:
            print(f"> Processing project {project} for {approach} approach.")
            return False


def tsa_model_demo(seasonality):
    """
    Executes the tsa process
    """

    # Check if Seasonality is taken into consideration
    if seasonality == True:
        output_directory = "sarimax_demo"
    else:
        output_directory = "arimax_demo"

    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data")
    output_path = os.path.join(DATA_PATH, output_directory)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "biweekly_results"))
    # List existing data files:
    biweekly_files = os.listdir(biweekly_data_path)
    assessment_statistics = ["PROJECT", "MAPE", "MSE", "MAE", "RMSE", "AIC", "BIC"]
    for i in tqdm(range(len(biweekly_files)), desc="Processing projects", unit="project"):

        project = biweekly_files[i][:-4]
        biweekly_results_path = os.path.join(output_path, "biweekly_results", f"{project}.csv")
        biweekly_assessment = pd.DataFrame(columns=assessment_statistics)

        # Check if the project has already been processed
        if detect_existing_output(project=project, paths=[biweekly_results_path],
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

        if seasonality:
            print(f"> SARIMAX modelling for project <{project}> performed - {i+1}/{len(biweekly_files)}")
        else:
            print(f"> ARIMAX modelling for project <{project}> performed - {i+1}/{len(biweekly_files)}")

    if seasonality:
        print("> SARIMAX stage performed!")
    else:
        print("> ARIMAX stage performed!")


def main():

    tsa_model_demo(seasonality=True)
    tsa_model_demo(seasonality=False)

if __name__ == '__main__':
    main()