<div align="center">

# [CovidCast: Predict to Protect](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Cast%20Final%20Presentation.pdf) 

<img src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="150" height="150"> 

## By Sam Celarek 
</div>


## 🎯 Project Overview
<div align="center" style="background-color: #f5f5f5; padding: 15px; border-radius: 10px;">

```
Like a weather forecast for pandemics,
COVIDCast leverages state-of-the-art machine learning and epidemiological models
to deliver precise outbreak predictions.
```
</div>

During COVID 19, the unknown course of the pandemic is almost as deadly as the virus itself. **COVIDCast** aims to address this problem by **Predicting to Protect**. To do this, COVIDCast utlizes the expert knowledge of epidemiological models and the forecasting power of times series models to predict the next 14 days of new COVID cases and protect governments, hospitals, and people from the coming storm. 

<img align="right" src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="200" height="200"> 

## 📊 Dataset
The data for this project comes from three sources: 

- [Google Open COVID Project](https://github.com/GoogleCloudPlatform/covid-19-open-data)
- [OWID COVID dataset](https://github.com/owid/covid-19-data)
- [CovsirPhy Library](https://github.com/lisphilar/covid19-sir/blob/main/README.md)

These repositories gathered data from a number of sources all over the world including the WHO, John Hopkins Hospital, and the CDC. I used 6 different csv's in the master file, combining together mobility data (Google), weather data (Google), government restrictions (Google), hospitalizations and tests (OWID), case counts and epidemiological variables (CovsirPhy). There was some feature overlap in the datasets from each source, but this was used to help impute missing data in the other datasets to create a more complete final [master_df](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Data/master_df.parquet).

## 🧹 [Data Cleaning](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/1.%20COVIDCast%20Preprocessing.ipynb)
Originally 30% of the data was missing. I used a variety of techniques to make this data more manageable:

- **Imputation**: Missing values in one feature would be imputed from another dataset with the same feature after checking for similarity (eg: `current_hospitalizations`, `fatal`). 
- **Interpolation**: Missing values within a continuous feature with an underlying exponential growth curve would be filled using polynomial order 2 interpolation (eg: `excess_mortality`, `derived_reproduction_rate`). 
- **Filling**: Missing values that were only updated if changed, or impossible before a certain date were filled forward or filled with zeroes respectively (eg: `vaccine_policy`, `new_vaccinations`)
- **Trimming the Horizons**: The horizons of my time series data was from 2020-02-15 until 2023-03-22, as many features didn't have values beyond these dates.
- **Dropping Columns:** Features that didn't have values up till 2023-03-22 were dropped (eg: the mobility and weather data).

## 👓 [EDA](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/2.%20COVIDCast%20EDA.ipynb)
To properly apply time series models to the data, I had to assess: 

- **Differencing**: Use various differencing orders to achieve stationarity, and as indicated by the PACF and ACF plots.
- **PACF and ACF**: Look at the PACF and ACF correllelograms to determine AR and MA orders.
- **Seasonality**: Apply seasonal decomposition and find the lag for seasonality
- **Target Normality**: Check the COVID Case numbers for normality and performed a BoxCox tranformation of the data

Here is my [Preprocessing and EDA Presentation](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Preprocessing%20and%20EDA.pdf)

## 💠 Modeling

COVIDCast works by taking Epidemiological SIRD model estimated parameters of spread, death, and recovery and plugging them into the time series models as exogenous variables to give the model better information about the underlying nature of the disease being predicted. 

### 🦠 Epidemiological Model Overview

- **[SIRD Model](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/1.%20COVIDCast%20Preprocessing.ipynb)**: The SIRD (Susceptible, Infected, Recovered, Deceased) model offers insights into real-time disease spread. It estimates the rate of change for different populations (susceptible, infected, recovered, and deceased) and computes the reproductive rate of the disease known as \( R_0 \) (pronounced 'R naught'). This was computed in the beginning of the Preprocessing Notebook using the CovsirPhy library.

### 🧪 Time Series Forecasting Algorithms

- [**ARIMA, SARIMA, SARIMAX models**](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/3.%20COVIDCast%20SARIMAX%20Model.ipynb): These models predict future trends using moving averages, linear autoregression, and differencing. They also incorporate seasonality and exogenous variables, making them potent tools for forecasting. However they don't model nonlinear trends very well and they require a lot of prior programming by the forecaster.

- [**Prophet model**](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/4.%20COVIDCast%20Prophet%20Model.ipynb): From Facebook's own description, "Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects." 
  

## 🔢 [Feature Selection and Tuning the Models](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/3.%20COVIDCast%20SARIMAX%20Model.ipynb):
For ARIMA time series models, I wanted to added exogenous variables that have information about how a target is going to fluctuate in the next time steps. To determine these features I had to assess each variables' Stationarity, Granger-Causality, Linear Correlation Strength, Multi-Collinearity, and Importance. I found that 6 variables I thought to be important due to background knowledge were actually the best additional regressors for the model. I used autoarima for grid searching SARIMAX's orders, and found the order of `(3,0,2) (2, 1, 1) [7] with intercept` to be the most predictive in cross-validation. 

For Prophet modeling, I grid searched with cross-validation over a range of 3 to 15 of the most important features as determined by Recursive Feature Elimination with LGMBoost as my regressor. The best performing model using this technique had 11 exogenous variables. I then grid searched over the hyperparameters to land on the final settings of `changepoint_prior_scale=10` , `seasonality_prior_scale=0.01`, `holidays_prior_scale=10`, and `growth='linear'`. 

### 📈 Results and Performance

Both models were trained on new cases of COVID from February 15th, 2020 to March 5th, 2023. The unseen test data from March 5th to March 21st, 2023 served as the final evaluation benchmark. 

Forecasts:

- SARIMAX Model: ![image](https://github.com/scelarek/Covid-Prediction-Capstone/assets/115444760/8e9aa4be-fcaf-43cc-9b12-67f45f422011)

- Prophet Model: ![image](https://github.com/scelarek/Covid-Prediction-Capstone/assets/115444760/2120bfe7-0951-4b47-a3b3-5869f514ac40)

- Testing metrics: ![image](https://github.com/scelarek/Covid-Prediction-Capstone/assets/115444760/6df1eefe-b840-40ea-9828-3c5a15eaa56f)

Here we can conclude that the SARIMAX model with 6 exogenous variables and `(3,0,2) (2, 1, 1) [7] with intercept` has best understood the underlying trend of the daily COVID cases. The literature on COVID case prediction suggest though that RNNs with LSTM's provide an even more accurate forecast with a sMAPE of 5%. As a continuation of this project I would like to develop my own RNN model to reproduce this result and explore if the SIRD model parameters contribute more its forecasting power.  

## 💡 Other Resources

Resources within this project

- **[Presentation of COVIDCast](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Cast%20Final%20Presentation.pdf)**
- **[Video of Final Presentation](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVIDcast_%20Predicting%20COVID%20Cases%20No%20Glasses%20Ad%20Lib%20(online-video-cutter.com).mp4)**
- **[Preprocessing and EDA Presentation](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Preprocessing%20and%20EDA.pdf)**
- **[Functions and Libraries](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/capstone_functions.py)**
- **[Main Clean Data File](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Data/master_df.parquet)**

Thank you for your interest in **COVIDCast**. For any further inquiries or insights, please feel free to reach out through the GitHub repository or at scelarek@gmail.com.

<div align="center">

**Best Wishes,**  
*Sam Celarek*

</div>

---
