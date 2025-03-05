# Main paths for the construction of the project

DATA_PATH = "/Users/mrobredo23/OULU/JSS-TDpred2024/data"
assessment_statistics = ["PROJECT", "MAPE", "MSE", "MAE", "RMSE"]
final_table_columns = ['Approach', 'Type', 'MAPE', 'MAE', 'MSE', 'RMSE']
FINAL_TS_TABLE_COLS = ["PROJECT", "MAPE", "MSE", "MAE", "RMSE", "AIC", "BIC"]
INITIAL_VARS = ['S1213', 'RedundantThrowsDeclarationCheck', 'S00117', 'S00122', 'S1488', 'S1905', 'UselessImportCheck',
                'DuplicatedBlocks', 'S1226', 'S00112', 'S1155', 'S00108', 'S1151', 'S1132', 'S1481']

# Flag values for process control of the project
PREPROCESSING = True
SARIMAX = True
RELATED_WORK = True
ML_MODELS = True
COMBINE_RESULTS = True
LONG_TERM_FORECASTING = True