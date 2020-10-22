import pandas as pd
import numpy as np
from utils import iterChunks
from datasets import getLoaderByName
from uuid import uuid4
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from tqdm import tqdm
import os
import json

EXPERIMENTS_RESULTS_PATH = 'experiments/results'

def loadDataset(selectedDataset):
    loader = getLoaderByName(selectedDataset)
    train, test = loader.load()
    train = train.rename(columns={ loader.valueColumn: 'value' })
    test = test.rename(columns={ loader.valueColumn: 'testValue' })
    train['time'] = train.index
    return train, test

def findExperiment(datasetName, algorithmName, hyperparams):
    experiments = getExperimentsTable()
    if not len(experiments): return
    experiments['hyperparams'] = experiments['hyperparams'].apply(lambda jsonParams: json.loads(jsonParams))
    queryResults = experiments[
        (experiments.datasetName == datasetName) &
        (experiments.algorithmName == algorithmName) &
        (experiments.hyperparams == hyperparams)
    ]
    if (len(queryResults)): return queryResults.iloc[0]

def saveResults(experimentId, data):
    if not os.path.isdir(EXPERIMENTS_RESULTS_PATH): os.makedirs(EXPERIMENTS_RESULTS_PATH)
    data.to_csv(f'{EXPERIMENTS_RESULTS_PATH}/{experimentId}.csv')

def getExperimentsTable():
    if os.path.isfile('./experiments.csv'):
        return pd.read_csv('./experiments.csv')
    else:
        df = pd.DataFrame()
        df.to_csv('./experiments.csv')
        return df

def naiveForecast(history, forecastHorizon):
    return np.repeat(history.values[-1], forecastHorizon)

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

def runExperiment(datasetName, algorithmName, hyperparams):
    train, test = loadDataset(datasetName)
    experiment = findExperiment(datasetName, algorithmName, hyperparams)
    if not experiment:
        experiment = pd.Series(dict(
            id=str(uuid4()),
            datasetName=datasetName,
            algorithmName=algorithmName,
            hyperparams=json.dumps(hyperparams)
        ))
        fullseries = train.copy()
        fullseries['trainValue'] = fullseries['value']
        saveResults(experiment['id'], fullseries)

    fullseries = pd.read_csv(f'{EXPERIMENTS_RESULTS_PATH}/{experiment["id"]}.csv')
    forecastHorizon = hyperparams['forecastHorizon']
    predict = predictFuncs[algorithmName]
    chunks = list(iterChunks(test, forecastHorizon))
    for i, chunk in enumerate(tqdm(chunks)):
        if len(fullseries) > len(train) + i*forecastHorizon: continue
        ypred = predict(fullseries, **hyperparams)
        newData = chunk.copy()
        newData['trainValue'] = newData.testValue
        newData[algorithmName] = ypred[:len(chunk)]
        fullseries = fullseries.append(newData)
        saveResults(experiment['id'], fullseries)
