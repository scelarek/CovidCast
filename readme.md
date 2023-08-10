<div align="center">

# [CovidCast](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Cast%20Final%20Presentation.pdf) 

<img src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="80" height="80"> 

## **COVIDCast: Predict to Protect**  
## **BrainStation**
</div>


## 🎯 Project Overview

Pandemics, even worse than COVID, are thought to be a major threat to the existence of humanity itself. **COVIDCast** aims to address this problem by Predicting to Protect. That is to say, it uses times series and epidemiological models to forecast the next 14 days of new COVID cases so governments, hospitals, and people can prepare for the coming storm. 

<img align="right" src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="400" height="400"> 

## 📊 Dataset

The data for this project comes from three sources: 

- [Google Open COVID Project](https://github.com/GoogleCloudPlatform/covid-19-open-data)
- [OWID COVID dataset](https://github.com/owid/covid-19-data)
- [COVSIRPHY Library](https://github.com/lisphilar/covid19-sir/blob/main/README.md)

These repositories gather data from a number of sources all over the world including the WHO, John Hopkins Hospital, and the CDC.

## [Data Cleaning](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/1.%20COVIDCast%20Preprocessing.ipynb)


**[Main Clean Data File](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Data/master_df.parquet)**


## [EDA](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/2.%20COVIDCast%20EDA.ipynb)
To properly apply time series models to the data, I had to: 
- Ensure stationarity using various differencing orders
- Check the COVID Case numbers for normality and performed a BoxCox tranformation of the data
- Assess the PACF and ACF correllelograms to determine AR and MA orders respectively
- Apply seasonal decomposition and find the lag for seasonality


## Feature Selection:
To determine which features I should include as exogenous variables in my models, I performed these steps:
- Assess linear correlation with the target variable
- Look at the nonlinear importance of different features using RFE with Random Forest Regression
- Assess Multi-Collinearity with the Variance Inflation Factor


**[Preprocessing and EDA Presentation](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Preprocessing%20and%20EDA.pdf)**


## Modeling

COVIDCast works by taking the output of the Epidemiological SIRD model and plugging that into the time series model to give it better information about the underlying nature of the disease being predicted. 

### 🦠 Epidemiological Model Overview

- **SIRD Model**: The SIRD (Susceptible, Infected, Recovered, Deceased) model offers insights into real-time disease spread. It estimates the rate of change for different populations (susceptible, infected, recovered, and deceased) and computes the reproductive rate of the disease known as \( R_0 \) (pronounced 'R naught'). This was computed in the beginning of the Preprocessing Notebook.

### 🧪 Time Series Forecasting Algorithms

- [**ARIMA, SARIMA, SARIMAX models**](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/3.%20COVIDCast%20SARIMAX%20Model.ipynb): These models predict future trends using moving averages, recent values, and differencing. They also incorporate seasonality and exogenous variables, making them potent tools for forecasting.

- [**Prophet model**](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/4.%20COVIDCast%20Prophet%20Model.ipynb): 
  

### 📈 Results and Performance

Models were trained on COVID data from February 15th, 2020 to March 5th, 2023. The unseen test data from March 5th to March 21st, 2023 served as the evaluation benchmark. Both SARIMAX and Prophet models were enhanced using features derived from the SIRD model.

## 💡 Other Resources

Explore various resources associated with this project:

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
