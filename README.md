# Journal of Systems and Software (JSS) submitted article : *"Evaluating Time-Dependent Methods and Seasonal Effects in Code Technical Debt Prediction"*

This replication package contains all the Python code to conduct the data collection, preprocessing and analysis of this study.

## Getting started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

Running the code requires Python3.9. See installation instructions [here](https://www.python.org/downloads/).
The dependencies needed to run the code are all listed in the file `requirements.txt`. They can be installed using pip:
```pip install -r requirements.txt```. Such requirements include classic statistics and ML libraries such as _scikit-learn_, _scipy_ or _statsmodels_ among others.

You might also want to consider using [virtual env](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/). Which we did.

## Structure of the replication package

The replication package provides two different directories. The first one provides the codes for running the study and the second one
provides the initial data files obtained from the dataset specified in the paper.

### Codes

The folder `../codes/` should contain the following files:
```commons.py```
```main.py```
```ml_modelling.py```
```ml_modelling_backward.py```
```modules.py```
```related_work.py```
```related_work_speed.py```
```result_combiner.py```
```ts_modelling.py```
```ts_modelling_speed.py```
```visualization.py```
```preprocessing.py```
```tsDataPreparation.py```

### Data
The folder `../data/` should contain the following files:
```biweekly_data/```
```monthly_data/```
```complete_data/```
```raw-data/```

Furthermore, if needed, the database file of the Technical Debt dataset is also provided in the replication package as ```td_V2.db``` file.





## Running the code

NOTE 1: Please, find the `DATA_PATH` global variable in the `commons.py` script and define the path where the program should create all the needed results.
The logic would be that you provide the base path is the location of this replication package in your machine, and you add `data` as the location for the data files.

NOTE 2: The different stages of the study execution are splitted in the ```main.py``` script, from the boolean definitions in
```commons.py``` practitioners can decide which stages want to be manipulated or re-executed again without affecting the other stages.
For a complete execution, set all the boolean global variables to ```True```

### Stage 1: ```PREPROCESSING```

- Executes scripts ```preprocessing.py``` and ```tsDataPreparation.py```.
- In these scripts the raw data from the Technical Debt dataset are converted into project-divided csv files repeated in 
```biweekly```, ```monthly``` and ```complete``` format by first performing data cleaning and preprocessing using the techniques
described in the paper.

### Stage 2: ```SARIMAX```

- Executes script ```ts_modelling_speed.py``` for executing the multivariate Time Series Analysis approach proposed:
  - ```seasonality=True```: Addresses the seasonality effect within the data through SARIMAX
  - ```seasonality=False```: Does not address the seasonality effect and implements just ARIMAX.

### Stage 3: ```RELATED_WORK```

- Executes script ```related_work_speed.py``` for executing the multivariate Time Series Analysis provided from the related work:
  - ```seasonality=True```: Addresses the seasonality effect within the data by implementing univariate SARIMA+LM.
  - ```seasonality=False```: Does not address the seasonality effect and implements just ARIMAX+LM.

### Stage 4: ```ML_MODELS```

- Executes script ```ml_modelling_backward.py``` for executing the considered ML models with backward variable selection procedure.

### Stage 5: ```COMBINE_RESULTS```

- Executes script ```result_combiner.py``` for combining the performance results from all the resulting projects.
  - Generates the final results from all the models by collecting their average results. The ```csv``` as well as ```LaTex``` tables can be found in ```../data/final_results/```.

### Stage 6: ```Visualization of the results```

- For the sake of flexibility, multiple visualization options apart from the ones displayed in the paper can be obtained by running all the cells existing in the Jupyter Notebook ```visualization.ipynb```.


### Stage 7: ```Long-term forecasting```

- Executes script ```revision_demo.py``` for performing long-term forecasting for both ARIMAX (with _biweekly_ data) ans SARIMAX (with _monthly_ data) given their results in the previous stages of the study. The multiple results should be directly generated 
in the locations `../data/arimax_demo` and `../data/sarimax_demo` accordingly.  



