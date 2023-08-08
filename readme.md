<div align="center">

# [CovidCast](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Cast%20Final%20Presentation.pdf) 

<img src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="80" height="80"> 

## **COVIDCast: Fusing Epidemiology and Time Series Forecasting to Predict Pandemics**  
## **BrainStation**



</div>


--- 

## 🌟 Introduction 

<div align="center">
Like a weather forecast for pandemics, COVIDCast leverages state-of-the-art machine learning and epidemiological models to deliver precise outbreak predictions.
</div>

<img align="right" src="https://github.com/scelarek/BrainStation_Capstone/blob/main/Presentations/Logo%20CovidCast.png?raw=true"  title="CovidCast" alt="CovidCast" width="400" height="400"> 

### Problem Statement
Three years ago, the emergence of COVID-19 drastically impacted the world. While COVID-19 was a significant challenge, the overarching threat of pandemics looms even larger. Experts from organizations like the WHO, alongside various independent research teams, have highlighted pandemics as one of the most significant threats to humanity. The statistics are daunting: there's a 1 in 30 chance that a pandemic could pose severe challenges to our existence in the upcoming century.

### My Mission

My goal with this project is straightforward: **Predict to Protect**. 

If we can anticipate the number of new cases, we stand a better chance of mitigating its impact. COVIDCast is unique, leveraging the principles of the epidemiological SIRD model combined with advanced time series forecasting algorithms like ARIMA, SARIMA, SARIMAX, and the Prophet model.

- **Time series forecasting algorithms:** The ARIMA and Prophet models predict future trends using the moving average, recent values, and differencing to predict the value of the target variable at the next time step. These models can also incorporate the information gained from seasonality and exogenous variables -- which is where the Epidemiological Models come in. 
- **Epidemiological Model Overview**: Traditional epidemiological container models, such as the SIRD (Susceptible, Infected, Recovered, Deceased), offer foundational insights into disease spread in real time. SIRD does this by estimating the most recent rate of change for for susceptible, infected, recovered, and deceased populations then using it to compute the reproductive rate of the disease called R0 (pronounced 'R not'). This reproductive rate is the main value that I will be plugging into my time series forecasting models as an exogenous variable.


My integrated approach captures the assumptions and knowledge of pandemics that experts have while using the power of time series models to predict seasonal trends and daily fluctuations.


### Results and Model Performance on Test Scores
The Time Series models were trained on COVID data from Februaury 15th 2020 to March 25th 2023, then tested on their results for the unseen test data 2023 1-25-2023 -> 2-09-2023)


SARIMAX and Prophet both enhanced by features derived from the SIRD model of the pandemics infectiousness. 

### Engage with Me and Future Prospects:
I encourage you to delve deeper into this repository. Should you have any inquiries or insights, please don't hesitate to reach out. 

---

### 🚀 Project Breakdown

My project is broken down into four main parts:

1. **[Cleaning](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/1.%20COVIDCast%20Preprocessing.ipynb)**: Dive into my process of data preprocessing and cleaning.
2. **[EDA](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/2.%20COVIDCast%20EDA.ipynb)**: Explore the basic exploratory data analysis I performed.
3. **[Initial ARIMA Modeling](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/3.%20COVIDCast%20SARIMAX%20Model.ipynb)**: See the first models I used to forecast COVID.
4. **[Facebook Prophet Modeling](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/4.%20COVIDCast%20Prophet%20Model.ipynb)**:  See the second model I used for forecasting.

---

### 💡 Resources

I have organized my functions, libraries, and primary clean data file for easy access:  

- **[Presentation of COVIDCast](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Cast%20Final%20Presentation.pdf):** This is the final presentation of COVIDCast.
- **[Video of Final Presentation](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVIDcast_%20Predicting%20COVID%20Cases%20No%20Glasses%20Ad%20Lib%20(online-video-cutter.com).mp4):** This is a video of me presenting COVIDCast.
- **[Presentation of Preprocessing and Cleaning](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Presentations/COVID%20Preprocessing%20and%20EDA.pdf):**  This is a presentation of the Preprocessing and EDA steps.
- **[Functions and Libraries](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Capstone/capstone_functions.py)**: Access the core functions and libraries used for this project.
- **[Main Clean Data File](https://github.com/scelarek/Covid-Prediction-Capstone/blob/main/Data/master_df.parquet)**: The primary cleaned data file.


---

I appreciate your interest in my project and invite you to dive deep into my work. If you have any questions or require further insights, please don't hesitate to reach out through this GitHub repository or at scelarek@gmail.com.

<div align="center">

**Best Wishes,**  
*Sam Celarek*

</div>

---
