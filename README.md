## COVID-19 Forecast: A Simple ARIMAX/ConvLSTM Comparison 

## Table of Contents
  * [Project Introduction](#project-introduction)
  * [Usage](#usage)
  * [Lab Notebook](#lab-notebook)
     * [11/28/2020](#11282020---2-hrs)
     * [12/01/2020](#12012020---2-hrs)
     * [12/03/2020](#12032020---4-hrs)


## Project Introduction

## Usage

## Lab Notebook

### 11/28/2020 - 2 hrs

- Created jupyter notebook for presentation
- Cleaned up datasets for the project
- Re-familiarized with Plotly and Pandas

### 12/01/2020 - 2 hrs

- Added function to convert dataset to time series data
- Added function to split dataset into train/validate/test partitions by state
- Added high level aggregate dataset processing function

### 12/03/2020 - 4 hrs
- Extracted and processed more state-related information for dataset
 construction. 
- Reconstruct data processing pipeline (RIP). ARIMA and LSTM are now taking
 differently structured datasets (same value).
- Redesigned LSTM model data: added nearest states as other independent
 variables, and added cases number for death predictions.
- Formalized pipeline: ARIMA vs. ConvLSTM per state, may use ensemble if the
 differences in predictions are not crazily large.
- TODO: build the actual models and design fine tuning pipeline for LSTM.

### To be Continued...
