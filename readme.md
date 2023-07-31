<div align="center">

# CovidCast 

<img src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="80" height="80"> 

## **COVIDCast: Fusing Epidemiology and Time Series Forecasting to Predict Pandemics**  
## **BrainStation**



</div>


--- 

## 🌟 Introduction 

<div align="center">
Like a weather forecast for pandemics, COVIDCast leverages state-of-the-art machine learning and epidemiological models to deliver precise outbreak predictions.
</div>


### Problem Statement
Three years ago, the emergence of COVID-19 drastically impacted the world. While COVID-19 was a significant challenge, the overarching threat of pandemics looms even larger. Experts from esteemed organizations like the WHO, alongside various independent research teams, have highlighted pandemics as one of the most significant threats to humanity. The statistics are daunting: there's a 1 in 30 chance that a pandemic could pose severe challenges to our existence in the upcoming century.

### My Mission

My goal with this project is straightforward: **Predict to Protect**. If we can anticipate the spread of a pandemic at its onset, we stand a better chance of mitigating its impact. COVIDCast is unique, leveraging the principles of the epidemiological SIRD model combined with advanced time series forecasting algorithms like ARIMA, SARIMA, SARIMAX, and the Prophet model.

- **Time series forecasting algorithms:** The ARIMA and Prophet models predict future trends using the moving average, recent values, and differencing to predict the value of the target variable at the next time step. These models can also incorporate the information gained from seasonality and exogenous variables -- which is where the Epidemiological Models come in. 
- **Epidemiological Model Overview**: Traditional epidemiological container models, such as the SIRD (Susceptible, Infected, Recovered, Deceased), offer foundational insights into disease spread in real time. SIRD does this by estimating the most recent rate of change for for susceptible, infected, recovered, and deceased populations then using it to compute the reproductive rate of the disease called R0 (pronounced 'R not'). This reproductive rate is the main value that I will be plugging into my time series forecasting models as an exogenous variable.

My integrated approach captures the intricacies of pandemics while maximizing the predictive capabilities of state-of-the-art statistical algorithms.

<img align="right" src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="400" height="400"> 

### Results and Model Performance
Two models emerged as particularly potent: SARIMAX and Prophet both enhanced by features derived from the SIRD's ODE model of the pandemics infectiousness. Below, you will find comprehensive evaluation statistics for these models, accompanied by visual representations showcasing their predictive prowess on unseen data.

### Engage with Me and Future Prospects:
I encourage you to delve deeper into this repository. Should you have any inquiries or insights, please don't hesitate to reach out. One of our forthcoming additions is an interactive notebook, enabling users to predict COVID-19 trends for any given day.

---

### 🚀 Project Breakdown

My project is broken down into four main parts:

1. **Cleaning**: Dive into my process of data preprocessing and cleaning in this [Jupyter Notebook](https://github.com/scelarek/BrainStation_Capstone/blob/d2dcb369dbfd98b2e8954b0028a0293529448294/Capstone/1.%20Covid%20Preprocessing.ipynb).
2. **EDA**: Explore the basic exploratory data analysis I performed in this [Jupyter Notebook](https://github.com/scelarek/BrainStation_Capstone/blob/d2dcb369dbfd98b2e8954b0028a0293529448294/Capstone/2.%20Sample%20EDA%20(Basic).ipynb).
3. **Initial ARIMA, SARIMA, and SARIMAX Modeling**: I performed in this [Jupyter Notebook](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/3.%20Covid%20SARIMA%20Modeling.ipynb)
4. **Facebook Prophet Modeling**:  I performed in this [Jupyter Notebook](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/4.%20Covid%20Prophet%20and%20RNNs.ipynb)

---

### 💡 Resources

I have organized my functions, libraries, and primary clean data file for easy access:  

- **Functions and Libraries**: Access the core functions and libraries used for this project in this [Python file](https://github.com/scelarek/BrainStation_Capstone/blob/d2dcb369dbfd98b2e8954b0028a0293529448294/Capstone/capstone_functions.py).
- **Main Clean Data File**: The primary cleaned data file is available in [this parquet file](https://github.com/scelarek/BrainStation_Capstone/blob/d2dcb369dbfd98b2e8954b0028a0293529448294/Data/master_df.parquet).
- **Presentation Preprocessing and Clean:**  This is a presentation of the [Preprocessing and EDA.](https://github.com/scelarek/BrainStation_Capstone/blob/e824c901efdb0adf1783256664bcfe054ae51001/Presentations/COVID%20Preprocessing%20and%20EDA.pdf)

---

I appreciate your interest in my project and invite you to dive deep into my work. If you have any questions or require further insights, please don't hesitate to reach out through this GitHub repository or at scelarek@gmail.com.

<div align="center">

**Best Wishes,**  
*Sam Celarek*

</div>

---
