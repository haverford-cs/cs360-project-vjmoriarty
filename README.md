## COVID-19 Forecast: A Simple ARIMAX/ConvLSTM Comparison 

## Table of Contents
  * [Project Introduction](#project-introduction)
  * [Usage](#usage)
  * [Lab Notebook](#lab-notebook)
     * [11/28/2020](#11282020---2-hrs)
     * [12/01/2020](#12012020---2-hrs)
     * [12/03/2020](#12032020---6-hrs)
     * [12/05/2020](#12052020---6-hrs)
     * [12/13/2020](#12132020---9-hrs)
     * [12/14/2020](#12142020---13-hrs)
     * [12/15/2020](#12152020---5-hrs)

## Project Introduction

This repo contains the python scripts and bash scripts implemented for the
 CS360 final project, along with some prediction results used for
  presentation. For a more comprehensive introduction to the whole project
   and pipeline design, see ```presentation.ipynb``` through jupyter notebook
    for more details.

---

## Usage

- Before running everything, make sure your working environment is set up
 properly. The scripts in this repo require ```python >= 3.6```. As for
  package, you can install all required packages by running:
  ```bash
    pip install -r requirements.txt
  ```

- Model configurations are stores at ```models/settings.py```. It contains
 general parameters for dataset generations, model specific parameters, and
  fine tuning related settings. Check out the script and make sure everything
   is as desired before running any models.
  
- To fine tune LSTM models, either run the bash script or the python script
 directly. All configurations are in ```models/settings.py```.
 
- To fine tune ARIMAX models, run
    ```bash
    bash fine_tune_arimax.sh
    ```
    - Some general settings, such as random search, are in ```models/settings.py```. 
    Other settings, such as random size and max time allowed for each
     hyper-parameter combination training, can be found in
      ```fine_tune_arimax.sh```. 

- To test models or use models to do rolling predictions, toggle the setting
 in  ```models/settings.py``` before running ```run_prediction.sh```.
 
- To see result visualization, open up the presentation jupyter notebook in
 the project and run all cells. The prediction plots are at the bottom in
  section 5.
---

## Lab Notebook

### 11/28/2020 - 2 hrs

- Created jupyter notebook for presentation
- Cleaned up datasets for the project
- Re-familiarized with Plotly and Pandas

### 12/01/2020 - 2 hrs

- Added function to convert dataset to time series data
- Added function to split dataset into train/validate/test partitions by state
- Added high level aggregate dataset processing function

### 12/03/2020 - 6 hrs
- Extracted and processed more state-related information for dataset
 construction. 
- Reconstruct data processing pipeline (RIP). ARIMA and LSTM are now taking
 differently structured datasets (same value).
- Redesigned LSTM model data: added nearest states as other independent
 variables, and added cases number for death predictions.
- Formalized pipeline: ARIMA vs. ConvLSTM per state, may use ensemble if the
 differences in predictions are not crazily large.

### 12/05/2020 - 6 hrs
- Fixed dataset problem with LSTM. 5-D data structure.
- Built a temporary LSTM architecture, tested with sample data
- Added basic ARIMAX training steps. 
- TODO: Architecture search and hyper-parameter tuning for LSTM
- TODO: ARIMAX full pipeline
- TODO: Prediction dataset and pipeline

### 12/13/2020 - 9 hrs
- Formalized model pipeline: 
    - Gather COVID data through local csv files
    - Convert dataset to time series format for both models
    - Fine tune both models and save the models through pickle
    - Run predictions through the better model by first unpack from pickle
     and then run predict
    - Pipelines above should be executed through two types of bash files
    : ```fine_tune.sh``` and ```run_prediction.sh```
- Established folder structure:
    - data: contains csv files, pickled dataset for models, info.py and dataset.py
    - models: contains ARIMAX and LSTM scripts, setting.py, and subfolder with
     pickled model parameters.
    - results: contains jupyter notebook, graphs, etc.
    - utils: utils.py, helper functions
- Added data cutoff and different offset settings for augmented deaths dataset.
- Completed LSTM fine tuning related scripts
    - NOTE: can't do architecture search within the time limit. Grid and
     architecture must change simultaneously.
- Fine tuning on lab machines. Fingers crossed.
- Completed ARIMAX train, validate, fine tuning.
- TODO: Run test data on both models
- TODO: Run future predictions with the better model.

### 12/14/2020 - 13 hrs
- Separated LSTM and ARIMAX fine tuning scripts.
- Added bash script for ARIMAX fine tuning to avoid convergence exception and
 long runtime. 
- ARIMAX and LSTM (with old architecture) random grid search fine tuning
 completed.
- Added test functions and rolling prediction functions. 
- Running test results and rolling predictions for ARIMAX on lab machines. 
Fingers crossed.
- TODO: Finished up interactive graphs and some part of the presentation.
- TODO: Run test data on both models once ARIMAX fine tuning is over.

### 12/15/2020 - 5 hrs
- Run test data on both models with fine tuned hyper-parameters
- Wrap up presentation and README.md

---

## References
* [COVID data from John Hopkins University](https://www.kaggle.com/antgoldbloom/covid19-data-from-john-hopkins-university)
* Documentations:
    * [Tensorflow Documentation on ConvLSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D)
    * [ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA)
    * [Plotly Documentation](https://plotly.com/python/)