"""
This project executes a series of TSA techniques as well as ML models

NOTE: The data collection and preprocessing has been already performed.
"""

import os
from ml_modelling import ml_models
#from ts_modelling import ts_models
#from related_work import related_models
from ts_modelling_speed import ts_models
from related_work_speed import related_models
from chart_creator import create_charts
from result_combiner import combine_results
from commons import SARIMAX, RELATED_WORK, ML_MODELS, COMBINE_RESULTS, VISUALIZE


def main():

    # SARIMAX modelling stage execution
    if SARIMAX:
        # ts_models(seasonality=True)
        ts_models(seasonality=False)  # Replicated work

    # SARIMA + LM related work stage execution
    if RELATED_WORK:
        related_models(seasonality=True)
        related_models(seasonality=False)  # Replicated work

    # ML stage
    if ML_MODELS:
        ml_models()

    # Combining model assessment results.
    if COMBINE_RESULTS:
        combine_results()

    # Make visualization of the results.
    if VISUALIZE:
        create_charts()


if __name__ == '__main__':
    main()