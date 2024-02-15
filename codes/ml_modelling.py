from modules import assessmentMetrics
from commons import DATA_PATH, assessment_statistics

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
import xgboost as xgb
import statsmodels.api as sm


def mlr_regression(training_df, testing_df, pro_name):
    """
    Performs the Multiple Linear Regression on the provided project.

    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """

    print(f"> Multiple Linear Regression for project {pro_name}")
    predictions = []

    X_train = training_df.iloc[:, 1:]
    y_train = training_df.iloc[:, 0]
    X_test = testing_df.iloc[:, 1:]
    y_test = testing_df.iloc[:, 0]

    for i in range(len(y_test)):

        # Explicitly adding the intercept of the Linear Regression
        X_train_constant = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train_constant).fit()

        # New observation for prediction
        new_observation = sm.add_constant(X_test.iloc[i:i+1, :], has_constant='add')
        prediction = model.predict(new_observation)
        predictions.append(prediction.values[0])

        # Update the training data
        X_train = pd.concat([X_train, X_test.iloc[i:i+1, :]])
        y_train = pd.concat([y_train, pd.Series(y_test.iloc[i])])

    print(f"> PROCESSED Multivariate Linear Regression for project {pro_name}")
    return assessmentMetrics(model='mlr', predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def svm_regression(training_df, testing_df, pro_name):
    """
    Performs the Support Vector Regression on the provided project.
    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """
    print(f"> Support Vector Machine Regression for project {pro_name}")

    # Empty array for the predictions
    predictions = []

    # Separating independent vars from dependent variable
    X_train = training_df.iloc[:, 1:]  # Independent variable
    y_train = training_df.iloc[:, 0]  # Dependent variable
    X_test = testing_df.iloc[:, 1:]
    y_test = testing_df.iloc[:, 0]

    for i in range(len(y_test)):

        # Pipeline to scale features and train the model
        model = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
        model.fit(X_train, y_train)

        # Make prediction for the next observation
        prediction = model.predict(X_test.iloc[i:i+1, :])
        predictions.append(prediction[0])

        X_train = pd.concat([X_train, X_test.iloc[i:i+1,:]])
        y_train = pd.concat([y_train, pd.Series(y_test.iloc[i])])

    print(f"> PROCESSED Support Vector Machine Regression for project {pro_name}")
    return assessmentMetrics(model="svr", predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def ridge_regression(training_df, testing_df, pro_name):
    """
    Performs the L2 Ridge regression on the provided project.

    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """
    print(f"> Ridge L2 Regression for project {pro_name}")

    # preparing the data
    X_train = training_df.iloc[:, 1:].values()  # Independent variable
    y_train = training_df.iloc[:, 0].values()  # Dependent variable
    X_test = testing_df.iloc[:, 1:].values()
    y_test = testing_df.iloc[:, 0].values()

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predictions = np.empty(len(X_test))

    for i in range(len(y_test)):
        model = Ridge(alpha=1.0, normalize=False, max_iter=10000)
        model.fit(X_train_scaled, y_train)

        prediction = model.predict(X_test_scaled[i].reshape[1,-1])
        predictions[i] = prediction

        # Expanding the training dataset for next loop
        X_train_scaled = np.vstack((X_train_scaled, X_test_scaled[i]))
        y_train = np.append(y_train, y_test[i])

    print(f"> PROCESSED L2 Ridge Regression for project {pro_name}")
    return assessmentMetrics(model="L2", predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def lasso_regression(training_df, testing_df, pro_name):
    """
    Performs the L1 Lasso regression on the provided project.
    :param training_df
    :param testing_df
    :param pro_name
    :return: Calls the assessment_metrics function with the obtained results
    """
    print(f"> Lasso L1 Regression for project {pro_name}")

    # preparing the data
    X_train = training_df.iloc[:, 1:].values()  # Independent variable
    y_train = training_df.iloc[:, 0].values()  # Dependent variable
    X_test = testing_df.iloc[:, 1:].values()
    y_test = testing_df.iloc[:, 0].values()

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predictions = np.empty(len(y_test))

    for i in range(len(X_test)):
        model = Lasso(alpha=1.0, normalize=False, max_iter=10000)
        model.fit(X_train_scaled, y_train)

        prediction = model.predict(X_test_scaled[i].reshape[1,-1])
        predictions[i] = prediction

        # Expanding the training dataset for next loop
        X_train_scaled = np.vstack((X_train_scaled, X_test_scaled[i]))
        y_train = np.append(y_train, y_test[i])

    print(f"> PROCESSED L1 Lasso Regression for project {pro_name}")
    return assessmentMetrics(model="L1", predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def xgboost(training_df, testing_df, pro_name):
    """
    Performs the XGBoost regression on the provided project.
    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """

    print(f"> XGBoost for project {pro_name}")
    predictions = []
    X_train = training_df.iloc[:, 1:].values()
    y_train = training_df.iloc[:, 0].values()
    X_test = testing_df.iloc[:, 1:].values()
    y_test = testing_df.iloc[:, 0].values()

    for i in range(len(y_test)):
        # Using DMatrix for train and test with xgb package.
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test[i].reshape(1, -1), label=y_test[i].reshape(-1))

        # Define model params
        params = {
            "max_depth": 3,
            "eta": 0.1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        }

        # Model metric
        watchlist = [(dtrain, "train"), (dtest, "test")]
        model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist, early_stopping_rounds=100, verbose_eval=False)

        prediction = model.predict(dtest, ntree_limit=model.best_ntree_limit)
        predictions.append(prediction[0])

        # Training data update
        X_train = np.vstack((X_train, X_test[i]))
        y_train = np.vstack((y_train, y_test[i]))

    print(f"> PROCESSED XGBoost for project {pro_name}")
    return assessmentMetrics(model="xgb", predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def rf_forest(training_df, testing_df, pro_name):
    """
    Performs the Random Forest Regression on the provided project.
    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """
    print(f"> Random Forest Regression for project {pro_name}")

    predictions = []
    # Separating independent vars from dependent variable
    X_train = training_df.iloc[:, 1:]  # Independent variable
    y_train = training_df.iloc[:, 0]  # Dependent variable
    X_test = testing_df.iloc[:, 1:]
    y_test = testing_df.iloc[:, 0]

    for i in range(len(y_test)):

        model = RandomForestRegressor(n_estimators=100, random_state=None, n_jobs=-1, bootstrap=False)
        model.fit(X_train, y_train)

        prediction = model.predict(X_test.iloc[i:i+1, :])
        predictions.append(prediction[0])

        X_train = pd.concat([X_train, X_test.iloc[i:i+1, :]])
        y_train = pd.concat([y_train, pd.Series(y_test.iloc[i])])

    print(f"> PROCESSED Random Forest Regression for project {pro_name}")
    return assessmentMetrics(model="rf", predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def sgd_regression(training_df, testing_df, pro_name):
    """
     Performs the Stochastic Gradient Descent Regression on the provided project.
    :param training_df:
    :param testing_df:
    :param pro_name:
    :return: Calls the assessment_metrics function with the obtained results
    """

    print(f"> Stochastic Gradient Descent for project {pro_name}")
    predictions = []
    scaler = StandardScaler()

    # Vectorized format for model features.
    X_train = training_df.iloc[:, 1:].values()
    y_train = training_df.iloc[:, 0].values()
    X_test = testing_df.iloc[:, 1:].values()
    y_test = testing_df.iloc[:, 0].values()

    # Scaling features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for i in range(len(y_test)):
        model = SGDRegressor(max_iter=1000, tol=0.0001)
        model.fit(X_train_scaled, y_train)

        prediction = model.predict(X_test_scaled[i].reshape(1, -1))
        predictions.append(prediction[0])

        X_train_scaled = np.vstack((X_train_scaled, X_test_scaled[i]))
        y_train = np.append(y_train, y_test[i])

    print(f"> PROCESSED Stochastic Gradient Descent for project {pro_name}")
    return assessmentMetrics(model="sgd", predicted_vals=predictions, testing_vals=y_test, pro_name=pro_name)


def ml_models():
    """
    Executes all the Ml models defined in the study
    :return:
    """

    biweekly_data_path = os.path.join(DATA_PATH, "biweekly_data")
    monthly_data_path = os.path.join(DATA_PATH, "monthly_data")
    complete_data_path = os.path.join(DATA_PATH, "complete_data")
    output_path = os.path.join(DATA_PATH, "results")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ml_results_path = os.path.join(output_path, "results_ML_sarimax")
    if not os.path.exists(ml_results_path):
        os.mkdir(ml_results_path)

    # data files
    biweekly_files = os.listdir(biweekly_data_path)
    monthly_files = os.listdir(monthly_data_path)
    complete_files = os.listdir(complete_data_path)

    ml_model_names = ["mlr", "svr", "L1", "L2", "xgb", "rf", "sgd"]
    periodicity_levels = ["monthly", "biweekly", "complete"]
    models = [mlr_regression, svm_regression, ridge_regression, lasso_regression, xgboost, rf_forest, sgd_regression]

    for i in range(len(models)):
        model_results_path = os.path.join(ml_results_path, ml_model_names[i])
        if os.path.exists(model_results_path):
            print("Results for ML model {} already processed".format(ml_model_names[i]))
            continue

        # Creation of nidek results direcotry
        os.mkdir(model_results_path)
        monthly_path = os.path.join(ml_results_path, f"{ml_model_names[i]}/monthly")
        biweekly_path = os.path.join(ml_results_path, f"{ml_model_names[i]}/biweekly")
        complete_path = os.path.join(ml_results_path, f"{ml_model_names[i]}/complete")

        os.mkdir(monthly_path)
        os.mkdir(biweekly_path)
        os.mkdir(complete_path)

        # Creation of the dataframe to hold the model assessment results
        monthly_assessment_df = pd.DataFrame(columns=assessment_statistics)
        biweekly_assessment_df = pd.DataFrame(columns=assessment_statistics)
        complete_assessment_df = pd.DataFrame(columns=assessment_statistics)

        for j in range(len(biweekly_files)):

            project_name = biweekly_files[j][:-4]  # Removes the .csv extension from the project name
            # Removing extra index number and commit_date columns from the original dataset
            biweekly_df = pd.read_csv(biweekly_files[j]).drop(columns=["COMMIT_DATE"])
            monthly_df = pd.read_csv(monthly_files[j]).drop(columns=["COMMIT_DATE"])
            complete_df = pd.read_csv(complete_files[j]).drop(columns=["COMMIT_DATE"])

            # Splitting the data into train set (80%) and test set (20%)
            biweekly_train_size = 0.8 * len(biweekly_df)
            monthly_train_size = 0.8 * len(monthly_df)
            complete_train_size = 0.8 * len(complete_df)

            biweekly_train = biweekly_df.iloc[:biweekly_train_size]
            monthly_train = monthly_df.iloc[:monthly_train_size]
            complete_train = complete_df.iloc[:complete_train_size]

            biweekly_test = biweekly_df.iloc[biweekly_train_size:]
            monthly_test = monthly_df.iloc[monthly_train_size:]
            complete_test = complete_df.iloc[complete_train_size:]

            # Model building & assessment
            biweekly_results = models[i](biweekly_train, biweekly_test, project_name)
            monthly_results = models[i](monthly_train, monthly_test, project_name)
            complete_results = models[i](complete_train, complete_test, project_name)

            # Updating biweekly results
            biweekly_assessment_df[j,:] = biweekly_results
            # Updating monthly results
            monthly_assessment_df[j,:] = monthly_results
            # updating complete results
            complete_assessment_df[j,:] = complete_results

        # Saving the results per periods in csv
        biweekly_assessment_df.to_csv(biweekly_path, index=False)
        monthly_assessment_df.to_csv(monthly_path, index=False)
        complete_assessment_df.to_csv(complete_path, index=False)