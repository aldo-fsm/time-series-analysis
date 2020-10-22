import pandas as pd
import numpy as np
from utils import iterChunks
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from tqdm import tqdm
import os
import json

def getExperiment(experimentId):
    experiments = getExperimentsTable()
    experiments[experimentId == experiments['id']].iloc[0]

def getExperimentsTable():
    if os.path.isfile('./experiments.csv'):
        return pd.read_csv('./experiments.csv')
    else:
        df = pd.DataFrame()
        df.to_csv('./experiments.csv')
        return df

def naiveForecast(history, forecastHorizon):
    return np.repeat(history[-1], forecastHorizon)

def simpleESForecast(history, forecastHorizon, alpha, trainingWindow=None):
    trainData = (history.tail(trainingWindow) if trainingWindow else history).trainValue
    exponentialSmoothing = SimpleExpSmoothing(trainData, alpha)
    exponentialSmoothing.fit(optimized=True)

    startIndex = len(trainData)
    endIndex = startIndex + forecastHorizon - 1
    return exponentialSmoothing.predict(exponentialSmoothing.params, start=startIndex, end=endIndex)

def holtWintersForecast(history, forecastHorizon, alpha, seasonal_periods, trend, seasonal, trainingWindow=None):
    trainData = (history.tail(trainingWindow) if trainingWindow else history).trainValue
    holt = ExponentialSmoothing(trainData, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    holt.fit()

    startIndex = len(trainData)
    endIndex = startIndex + forecastHorizon - 1
    return holt.predict(holt.params, start=startIndex, end=endIndex)

NAIVE='naive'
HOLT_WINTERS='holtWinters'
EXPONENTIAL_SMOOTHING='exponentialSmoothing'

predictFuncs = {
    NAIVE: naiveForecast,
    EXPONENTIAL_SMOOTHING: simpleESForecast,
    HOLT_WINTERS: holtWintersForecast,
}

def runExperiment(experimentId, trainSize, test, algorithmName, hyperparams):
    experiment = getExperiment(experimentId)
    fullseries = pd.read_csv(experiment['resultsTable'])
    forecastHorizon = hyperparams['forecastHorizon']
    predict = predictFuncs[algorithmName]
    chunks = list(iterChunks(test, forecastHorizon))
    for i, chunk in tqdm(enumerate(chunks)):
        if not pd.isna(fullseries.iloc[trainSize + i*forecastHorizon].trainValue): continue
        ypred = predict(fullseries, **hyperparams)
        newData = chunk.copy()
        newData['trainValue'] = newData.testValue
        newData[algorithmName] = ypred[:len(chunk)]
        fullseries = fullseries.append(newData)

