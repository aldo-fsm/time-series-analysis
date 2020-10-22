import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datasets import spxLoader, DATASET_NAMES, getLoaderByName, loadDataFrame
from utils import iterChunks
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error

selectedDataset = st.selectbox('Selecione a dataset', DATASET_NAMES)

@st.cache()
def load(selectedDataset):
    loader = getLoaderByName(selectedDataset)
    train, test = loader.load()
    return loader, train, test

loader, train, test = load(selectedDataset)
train = train.rename(columns={ loader.valueColumn: 'value' })
test = test.rename(columns={ loader.valueColumn: 'testValue' })

train['time'] = train.index

# exponentialSmoothing = SimpleExpSmoothing(train.value)
# exponentialSmoothing.fit()

initialTrainWindowSize = 100
trainWindowSize = st.number_input('Janela de treinamento', value=initialTrainWindowSize, min_value=1, max_value=len(train), step=1)

initialPeriod=25
period = st.number_input('Periodo para sasonalidade', value=initialPeriod, min_value=1, max_value=trainWindowSize, step=1)

initialForecastHorizon = 25
forecastHorizon = st.number_input('Horizonte de previsão', value=initialForecastHorizon, min_value=1, max_value=len(train), step=1)

progress = st.progress(0)
@st.cache()
def forecast(train, test, forecastHorizon, period, trainWindowSize):
    fullSeries = train.copy()
    fullSeries['trainValue'] = fullSeries.value

    chunks = list(iterChunks(test, forecastHorizon))
    for i, chunk in enumerate(chunks):
        trainData = fullSeries.tail(trainWindowSize).trainValue
        exponentialSmoothing = SimpleExpSmoothing(trainData)
        exponentialSmoothing.fit()

        holt = ExponentialSmoothing(trainData, trend='add', seasonal='add', seasonal_periods=period)
        holt.fit()

        startIndex = len(trainData) #len(fullSeries)
        endIndex = startIndex + forecastHorizon - 1
        predExpsmo = exponentialSmoothing.predict(exponentialSmoothing.params, start=startIndex, end=endIndex)
        predHolt = holt.predict(holt.params, start=startIndex, end=endIndex)

        print(len(predExpsmo), len(predHolt))
        newData = chunk.copy()
        print(chunk.shape)
        newData['trainValue'] = newData.testValue
        newData['expsmo'] = predExpsmo[:len(chunk)]
        newData['holt'] = predHolt[:len(chunk)]
        print(newData)
        fullSeries = fullSeries.append(newData)
        print(len(fullSeries))
        progress.progress((i+1)/(len(chunks)))
    return fullSeries

fullSeries = forecast(train, test, forecastHorizon, period, trainWindowSize)
test = fullSeries.tail(len(test))
test
fullSeries.shape
st.write("Predição")
st.write(
    px.line(fullSeries, x=fullSeries.index, y=['value', 'expsmo', 'holt', 'testValue'])
)

errors = pd.DataFrame()
errors['holtSquaredError'] = (test.testValue - test.holt)**2
errors['expsmoSquaredError'] = (test.testValue - test.expsmo)**2

st.write("Erro quadrático médio")
st.write(
    pd.DataFrame({
        'MSE': [
            mean_squared_error(test.testValue.values, test.holt.values),
            mean_squared_error(test.testValue.values, test.expsmo.values)
        ],
    }, index=['Holt', 'Exponential Smoothing'])
)

st.write("Erros quadráticos")
st.write(
    errors
)

initialWindow = int(len(test)*0.01)
window = st.number_input('Janela de analise do erro', value=initialWindow, min_value=1, max_value=len(errors), step=1)

st.write("Gráfico dos erros")
holtRollingError = errors.holtSquaredError.rolling(window)
expsmoRollingError = errors.expsmoSquaredError.rolling(window)
errors['holtMeanSquaredError'] = holtRollingError.mean()
errors['expsmoMeanSquaredError'] = expsmoRollingError.mean()
st.write(
    px.line(errors, y=['holtSquaredError', 'expsmoSquaredError', 'holtMeanSquaredError', 'expsmoMeanSquaredError'])
)
