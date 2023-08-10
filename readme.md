<div align="center">

# [CovidCast](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Cast%20Final%20Presentation.pdf) 

<img src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="80" height="80"> 

## **COVIDCast: Predict to Protect**  
## **BrainStation**
</div>


## 🎯 Project Overview

During COVID 19, the unknown course of the pandemic is almost as deadly as the virus itself. **COVIDCast** aims to address this problem by Predicting to Protect. That is to say, it uses times series and epidemiological models to forecast the next 14 days of new COVID cases so governments, hospitals, and people can prepare for the coming storm. 

<img align="right" src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="400" height="400"> 

## 📊 Dataset

The data for this project comes from three sources: 

- [Google Open COVID Project](https://github.com/GoogleCloudPlatform/covid-19-open-data)
- [OWID COVID dataset](https://github.com/owid/covid-19-data)
- [COVSIRPHY Library](https://github.com/lisphilar/covid19-sir/blob/main/README.md)

These repositories gathered data from a number of sources all over the world including the WHO, John Hopkins Hospital, and the CDC. I used 6 different csv's in the master file, combining together mobility data (Google), weather data (Google), government restrictions (Google), hospitalizations and tests (OWID), case counts and epidemiological variables (CovsirPhy). There was some overlap in the datasets from each source, but this was used to help impute missing data in the other datasets to create a more complete final master_df.

**[Main Clean Data File](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Data/master_df.parquet)**

## [Data Cleaning](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/1.%20COVIDCast%20Preprocessing.ipynb)
Originally 30% of the data was missing. I used a variety of techniques to make this data more manageable:
- **Imputation**: Missing values in one column would be imputed from another column if they had the same information (eg: current hospitalizations, fatalities). 
- **Interpolation**: Missing values within a continuous feature with an underlying exponential growth curve would be interpolated using polynomial order 2 (eg: excess mortality, SIRD variables). 
- **Forward Filling**: Missing values that were only updated if changed were filled forward, and assumed to be zero before the first value (eg: vaccine policy, school closings)
- **Filling with Zeros**: Missing values for data that would be zero because it wasn't possible until a certain date would be filled with zeros (eg: vaccines pre-2020 September).
- **Trimming the Horizons**: Many variables were not available until 2020-02-15 (current hospitalizations) or after 2023-03-22 (fatalities), therefore I used this as my start date.  


## [EDA](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/2.%20COVIDCast%20EDA.ipynb)
To properly apply time series models to the data, I had to assess: 
- **Stationarity**: Ensure stationarity using various differencing orders
- **Normality**: Check the COVID Case numbers for normality and performed a BoxCox tranformation of the data
- **PACF and ACF**: Look at the PACF and ACF correllelograms to determine AR and MA orders respectively
- **Seasonality**: Apply seasonal decomposition and find the lag for seasonality


## Feature Selection:
To determine which features I should include as exogenous variables in my models, I performed these steps:
- **Linear Correlation**: Assess linear correlation with the target variable
- **Importance**: Look at the importance of different features using Recursive Feature Elimination with Random Forest Regression
- **Multi-Collinearity**: Assess Multi-Collinearity with the Variance Inflation Factor


**[Preprocessing and EDA Presentation](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Preprocessing%20and%20EDA.pdf)**


## Modeling

COVIDCast works by taking the output of the Epidemiological SIRD model and plugging that into the time series model to give it better information about the underlying nature of the disease being predicted. 

### 🦠 Epidemiological Model Overview

- **SIRD Model**: The SIRD (Susceptible, Infected, Recovered, Deceased) model offers insights into real-time disease spread. It estimates the rate of change for different populations (susceptible, infected, recovered, and deceased) and computes the reproductive rate of the disease known as \( R_0 \) (pronounced 'R naught'). This was computed in the beginning of the Preprocessing Notebook using the COVSIRPHY library.

### 🧪 Time Series Forecasting Algorithms

- [**ARIMA, SARIMA, SARIMAX models**](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/3.%20COVIDCast%20SARIMAX%20Model.ipynb): These models predict future trends using moving averages, recent values, and differencing. They also incorporate seasonality and exogenous variables, making them potent tools for forecasting.

- [**Prophet model**](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/4.%20COVIDCast%20Prophet%20Model.ipynb): 
  

### 📈 Results and Performance

Models were trained on COVID data from February 15th, 2020 to March 5th, 2023. The unseen test data from March 5th to March 21st, 2023 served as the evaluation benchmark. Both SARIMAX and Prophet models were enhanced using features derived from the SIRD model.

## 💡 Other Resources

Popular resources within this project

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
