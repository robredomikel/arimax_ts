"""
Performs the predictions with the selected models ARIMAX and SARIMAX with a 70/30 split for survey demo to practitioners
"""

from commons import DATA_PATH
from modules import (MAPE, RMSE, MAE, MSE, check_encoding, create_diagnostics,
                     absolute_error, squared_error, sape_error)
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import pandas as pd
from csv import writer
from tqdm import tqdm
import numpy as np
from ts_modelling_speed import assessment_metrics, backward_modelling
import json
import matplotlib.pyplot as plt
import seaborn as sns


def detect_existing_output(project, paths, flag_num, files_num, approach):

    biweekly_results_path = paths[0]
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


def demo_testing(training_df, testing_df, best_model_cfg, best_regressors, seasonality, project):
    """
    Given the best model order parameters obtained from the backward variable selection and aut_arima tuning we
    fit the SARIMAX model for forecasting
    """

    arima_order = best_model_cfg[0]
    s_order = best_model_cfg[1]

    # Training the SARIMAX model
    existing_columns = list(set(best_regressors).intersection(training_df.columns))
    X_train = training_df[existing_columns].astype(float)
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

    predictions = []
    confidence_intervals = []

    for i in range(len(testing_df)):

        # Model forecasting
        best_reg_df = testing_df[existing_columns]
        best_reg_df_scaled = np.log1p(best_reg_df)
        X_test = best_reg_df_scaled.iloc[i, :].values.reshape(1, -1)
        y_pred = fitted_model.get_forecast(steps=1, exog=X_test)
        predictions.append(y_pred.predicted_mean[0])
        confidence_intervals.append(y_pred.conf_int(alpha=0.05)[0])

    print(f"> Model testing stage performed for project {project}")
    return predictions, confidence_intervals, round(fitted_model.aic, 2), round(fitted_model.bic, 2)


def generate_lineplot(real_vls_array, predicted_vals, conf_intervals, project_name, output_path,
                      seasonality):

    # Generate Line Plot
    plt.figure(figsize=(10, 6))
    x_range = range(len(real_vls_array))
    plt.plot(x_range, real_vls_array, label="Observed", color="blue", marker="o")
    plt.plot(x_range, predicted_vals, label="Predicted", color="orange", linestyle="--", marker="x")
    lower_bounds = [ci[0] for ci in conf_intervals]
    upper_bounds = [ci[1] for ci in conf_intervals]
    plt.fill_between(x_range, lower_bounds, upper_bounds, color="gray", alpha=0.3, label="95% CI")
    if seasonality:
        plt.title(f"Prediction vs Observed for Project {project_name} - SARIMAX")
        plt.xlabel("Months")
    else:
        plt.title(f"Prediction vs Observed for Project {project_name} - ARIMAX")
        plt.xlabel("Biweekly periods")
    plt.ylabel("SQALE INDEX")
    plt.legend()
    plt.grid()
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)
    plot_path = os.path.join(figures_path, f"{project_name}_lineplot.pdf")
    plt.savefig(plot_path)
    plt.close()
    print(f"> Line plot saved at {plot_path}")


def point_assessment(project_name, predicted_vals, conf_intervals, real_values, seasonality):
    """
    Monitors the point accuracy to understand the impact of the prediction length window
    """

    csv_header = ["id", "observed_value", "predicted_value", "lower_95", "upper_95", "abs_error", "sq_error", "sAPE_error",
                  "mape", "mae", "rmse", "mse"]
    if seasonality:
        output_path = os.path.join(DATA_PATH, "sarimax_demo")
    else:
        output_path = os.path.join(DATA_PATH, "arimax_demo")
    point_output_path = os.path.join(output_path, "point_assessment")
    os.makedirs(point_output_path, exist_ok=True)
    csv_file_path = os.path.join(point_output_path, f"{project_name}.csv")

    with open(csv_file_path, "w", newline="") as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(csv_header)
        f_object.close()

    real_vls_array = real_values.to_numpy()
    i=0
    for pred, real, conf in zip(np.asarray(predicted_vals), real_vls_array, conf_intervals):

        lower_95, upper_95 = conf
        abs_error = absolute_error(pred=pred, obvs=real)
        sq_error = squared_error(pred=pred, obvs=real)
        sp_error = sape_error(pred=pred, obvs=real)
        mape = MAPE(predicted_vals=predicted_vals[0:i+1], testing_vals=real_vls_array[0:i+1])
        mae = MAE(predicted_vals=predicted_vals[0:i+1], testing_vals=real_vls_array[0:i+1])
        rmse = RMSE(predicted_vals=predicted_vals[0:i+1], testing_vals=real_vls_array[0:i+1])
        mse = MSE(predicted_vals=predicted_vals[0:i+1], testing_vals=real_vls_array[0:i+1])

        with open(csv_file_path, "a", newline="") as f:
            writer_object = writer(f)
            writer_object.writerow([i, real, pred, lower_95, upper_95, abs_error, sq_error, sp_error, mape, mae, rmse, mse])
            f_object.close()
        i+=1

        generate_lineplot(real_vls_array, predicted_vals, conf_intervals, project_name, output_path, seasonality)


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
    split_point = round(len(sqale_index)*0.8)  # Initial data splitting. (70% training 30% testing)
    training_df = df.iloc[:split_point, :]
    testing_df = df.iloc[split_point:, :]

    """
    # SARIMAX backward modelling
    best_model_params, best_aic, best_regressors, output_flag = backward_modelling(df=training_df,
                                                                                   periodicity=periodicity,
                                                                                   seasonality=seasonality,
                                                                                   output_flag=True)

    # If there was error in the calculation of the backward modelling then we exclude the results from this project
    # for the given data format.
    if output_flag is False:
        return [project_name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    """

    print(f"PROJECT NAME: {project_name}")
    if seasonality:
        model_path = os.path.join(DATA_PATH, "best_sarimax_models/monthly", f"{project_name}.json")
    else:
        model_path = os.path.join(DATA_PATH, "best_arimax_models/biweekly", f"{project_name}.json")

    if not os.path.exists(model_path):
        return [project_name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    with open(model_path, "r", encoding="utf-8") as f_json:
        json_data = json.load(f_json)

    best_model_params = json_data["model_params"]
    best_regressors = json_data["best_regressors"]

    # Model testing
    try:

        # Creating model diagnostics for visualization:
        #create_diagnostics(seasonality, periodicity, best_model_params, best_regressors, training_df, project_name,
        #                   demo=True)
        predictions, conf_interval, aic_val, bic_val = demo_testing(training_df=training_df, testing_df=testing_df,
                                                                    best_model_cfg=best_model_params, best_regressors=best_regressors,
                                                                    seasonality=seasonality, project=project_name)

        point_assessment(project_name=project_name, predicted_vals=predictions, conf_intervals=conf_interval,
                         real_values=testing_df["SQALE_INDEX"], seasonality=seasonality)
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


def generate_project_normalized_mape_boxplots(output_path, seasonality):
    """
    Generate boxplots for MAPE values per biweekly observation across all projects.
    Normalize MAPE values by the number of analyzed repositories,
    dynamically adjusting for projects that stop contributing data.
    """

    # Define paths
    results_dir = os.path.join(output_path, "point_assessment")

    # Collect MAPE data
    mape_data = []
    project_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

    project_observations = {}
    for file in project_files:
        project_path = os.path.join(results_dir, file)
        project_df = pd.read_csv(project_path)
        project_name = file[:-4]
        project_observations[project_name] = project_df["mape"].tolist()

    # Find the maximum number of observations
    max_observations = max(len(mape_values) for mape_values in project_observations.values())

    # Set the display limit based on seasonality
    display_limit = 36 if seasonality else 72
    max_observations = min(max_observations, display_limit)

    # Normalize MAPE values
    normalized_mape_data = []
    for observation in range(max_observations):
        active_projects = 0
        observation_values = []
        for project, mape_values in project_observations.items():
            if observation < len(mape_values):  # Ensure the project has data for this observation
                observation_values.append(mape_values[observation])
                active_projects += 1

        if active_projects > 0:  # Avoid division by zero
            normalized_observation_values = [value / active_projects for value in observation_values]
            for value in normalized_observation_values:
                normalized_mape_data.append({"Observation": observation + 1, "Normalized MAPE": value})

    # Create a DataFrame for visualization
    normalized_mape_df = pd.DataFrame(normalized_mape_data)

    # Plot boxplots for each observation
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Observation", y="Normalized MAPE", data=normalized_mape_df, palette="Blues", showfliers=False)
    if seasonality:
        plt.title("Normalized MAPE Boxplot Per Monthly Observation Across Projects - SARIMAX")
        plt.xlabel("Monthly Observation Index")
    else:
        plt.title("Normalized MAPE Boxplot Per Biweekly Observation Across Projects - ARIMAX")
        plt.xlabel("Biweekly Observation Index")

    plt.ylabel("Normalized MAPE")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and show the plot
    plot_path = os.path.join(output_path, "normalized_project_mape_boxplot.pdf")
    plt.savefig(plot_path)
    plt.show()
    print(f"> Normalized MAPE boxplot saved at {plot_path}")


def generate_mape_boxplots(output_path, seasonality):
    """
    Generate boxplots for MAPE values per biweekly observation across all projects.
    """

    # Define paths
    if seasonality:
        results_dir = os.path.join(output_path, "point_assessment")
    else:
        results_dir = os.path.join(output_path, "point_assessment")

    # Collect MAPE data
    mape_data = []
    project_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

    for file in project_files:
        project_path = os.path.join(results_dir, file)
        project_df = pd.read_csv(project_path)

        # Extract biweekly observation indices and MAPE values
        for i, row in project_df.iterrows():
            mape_data.append({"Observation": i, "MAPE": row["mape"], "Project": file[:-4]})

    # Create a DataFrame for visualization
    mape_df = pd.DataFrame(mape_data)

    # Plot boxplots for each observation
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Observation", y="MAPE", data=mape_df, palette="Blues", showfliers=False)
    if seasonality:
        plt.title("MAPE Boxplot Per Monthly Observation Across Projects - SARIMAX")
        plt.xlabel("Monthly Observation Index")
    else:
        plt.title("MAPE Boxplot Per Biweekly Observation Across Projects - ARIMAX")
        plt.xlabel("Biweekly Observation Index")

    plt.ylabel("MAPE")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and show the plot
    if seasonality:
        plot_path = os.path.join(output_path, "mape_boxplot.pdf")
    else:
        plot_path = os.path.join(output_path, "mape_boxplot.pdf")
    plt.savefig(plot_path)
    plt.show()
    print(f"> Boxplot saved at {plot_path}")


def generate_mini_mape_boxplot(output_path, seasonality):
    """
    Generate boxplots for MAPE values per biweekly observation across all projects in a diminished format.
    """

    # Define paths
    if seasonality:
        results_dir = os.path.join(output_path, "point_assessment")
    else:
        results_dir = os.path.join(output_path, "point_assessment")

    # Collect MAPE data
    mape_data = []
    project_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    min_observations = float("inf")

    for file in project_files:
        project_path = os.path.join(results_dir, file)
        project_df = pd.read_csv(project_path)

        min_observations = min(min_observations, len(project_df))
        # Extract biweekly observation indices and MAPE values
        for i, row in project_df.iterrows():
            if i < min_observations:
                mape_data.append({"Observation": i, "MAPE": row["mape"], "Project": file[:-4]})

    # Create a DataFrame for visualization
    mape_df = pd.DataFrame(mape_data)
    mape_df = mape_df[mape_df["Observation"] < min_observations]

    # Plot boxplots for each observation
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Observation", y="MAPE", data=mape_df, palette="Blues", showfliers=False)
    if seasonality:
        plt.title("MAPE Boxplot Per Monthly Observation Across Projects - SARIMAX")
        plt.xlabel("Monthly Aggregated MAPE results")
    else:
        plt.title("MAPE Boxplot Per Biweekly Observation Across Projects - ARIMAX")
        plt.xlabel("Biweekly Aggregated MAPE results")

    plt.ylabel("MAPE")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and show the plot
    if seasonality:
        plot_path = os.path.join(output_path, "cropped_mape_boxplot_sarimax.pdf")
    else:
        plot_path = os.path.join(output_path, "cropped_mape_boxplot_arimax.pdf")
    plt.savefig(plot_path)
    plt.show()
    print(f"> Boxplot saved at {plot_path}")


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

    # Generate boxplot for MAPE values
    generate_mape_boxplots(output_path, seasonality)
    generate_mini_mape_boxplot(output_path, seasonality)
    generate_project_normalized_mape_boxplots(output_path, seasonality)


def analyze_distance_and_plot(output_path, seasonality):
    """
    Analyze the distances (absolute errors) and generate plots for all projects.
    """

    # Define the directory containing point assessment results
    if seasonality:
        results_dir = os.path.join(output_path, "point_assessment")
    else:
        results_dir = os.path.join(output_path, "point_assessment")

    project_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

    # Gather distance (abs_error) data across all projects
    distances = {}
    max_observations = float("inf")

    for file in project_files:
        project_path = os.path.join(results_dir, file)
        project_df = pd.read_csv(project_path)

        project_name = file[:-4]
        distances[project_name] = project_df["abs_error"].to_list()

        # Find the minimum number of observations across projects
        max_observations = min(max_observations, len(distances[project_name]))

    # Trim data to the minimum number of observations
    for project in distances:
        distances[project] = distances[project][:max_observations]

    # Scatterplot for each project
    for project, errors in distances.items():
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(errors)), errors, label=f"{project} Distances", color="orange", alpha=0.7)
        if seasonality:
            plt.title(f"Observed vs Predicted Distance for {project} - SARIMAX")
            plt.xlabel("Monthly observation index")
        else:
            plt.title(f"Observed vs Predicted Distance for {project} - ARIMAX")
            plt.xlabel("Biweekly observation index")

        plt.ylabel("Distance (In minutes to remediate Code TD)")
        plt.grid(alpha=0.5)
        figures_path = os.path.join(output_path, "figures", "distance_scatterplots")
        os.makedirs(figures_path, exist_ok=True)
        plt.savefig(os.path.join(figures_path, f"{project}_distance_scatterplot.pdf"))
        plt.close()
        print(f"> Scatterplot for project {project} saved.")

    # Calculate mean distances across all projects
    mean_distances = [sum(errors[i] for errors in distances.values()) / len(distances) for i in range(max_observations)]

    # Scatterplot for mean distances
    plt.figure(figsize=(10, 6))
    plt.scatter(range(max_observations), mean_distances, label="Mean Distance Across Projects", color="green",
                alpha=0.8)
    plt.title("Mean Observed vs Predicted Distance Across Projects")
    if seasonality:
        plt.xlabel("Monthly observation Index")
    else:
        plt.xlabel("Biweekly observation Index")
    plt.ylabel("Mean (In minutes to remediate Code TD)")
    plt.grid(alpha=0.5)
    mean_plot_path = os.path.join(output_path, "mean_distance_scatterplot.pdf")
    plt.savefig(mean_plot_path)
    plt.close()
    print(f"> Mean distance scatterplot saved at {mean_plot_path}")


def main():

    tsa_model_demo(seasonality=True)
    tsa_model_demo(seasonality=False)
    #analyze_distance_and_plot(output_path=os.path.join(DATA_PATH, "sarimax_demo"), seasonality=True)
    #analyze_distance_and_plot(output_path=os.path.join(DATA_PATH, "arimax_demo"), seasonality=False)


if __name__ == '__main__':
    main()