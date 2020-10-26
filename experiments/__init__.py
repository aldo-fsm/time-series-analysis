import pandas as pd
import numpy as np
from utils import iterChunks
from utils.parsers import parseDataset
from datasets import getLoaderByName
from uuid import uuid4
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from tqdm import tqdm
import os
import json

EXPERIMENTS_RESULTS_PATH = 'experiments/results'
EXPERIMENTS_TABLE_PATH = 'experiments/experiments.csv'

def loadDataset(selectedDataset):
    loader = getLoaderByName(selectedDataset)
    train, test = loader.load()
    train = train.rename(columns={ loader.valueColumn: 'value' })
    test = test.rename(columns={ loader.valueColumn: 'testValue' })
    train['time'] = train.index
    test['time'] = test.index
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

def appendResults(experimentId, data):
    if not os.path.isdir(EXPERIMENTS_RESULTS_PATH): os.makedirs(EXPERIMENTS_RESULTS_PATH)
    resultsTablePath = f'{EXPERIMENTS_RESULTS_PATH}/{experimentId}.csv'
    if os.path.isfile(resultsTablePath):
        data.to_csv(resultsTablePath, index=False, mode='a', header=False)
    else:
        data.to_csv(resultsTablePath, index=False)

def getExperimentsTable():
    if os.path.isfile(EXPERIMENTS_TABLE_PATH):
        return pd.read_csv(EXPERIMENTS_TABLE_PATH)
    else:
        df = pd.DataFrame(columns=[
            "id",
            "datasetName",
            "algorithmName",
            "hyperparams",
        ])
        df.to_csv(EXPERIMENTS_TABLE_PATH, index=False)
        return df

def saveNewExperiment(experiment):
    experiment = pd.DataFrame([experiment])
    if os.path.isfile(EXPERIMENTS_TABLE_PATH):
        experiment.to_csv(EXPERIMENTS_TABLE_PATH, mode='a', header=False)
    else:
        experiment.to_csv(EXPERIMENTS_TABLE_PATH)

def findOrCreateExpeiment(datasetName, algorithmName, hyperparams={}):
    train, test = loadDataset(datasetName)
    experiment = findExperiment(datasetName, algorithmName, hyperparams)
    if experiment is None:
        experiment = pd.Series(dict(
            id=str(uuid4()),
            datasetName=datasetName,
            algorithmName=algorithmName,
            hyperparams=json.dumps(hyperparams)
        ))
        saveNewExperiment(experiment)
    return experiment

def getHyperparams(experiment):
    hyperparams = experiment['hyperparams']
    return json.loads(hyperparams) if type(hyperparams) == str else hyperparams

def naiveForecast(history, forecastHorizon):
    return np.repeat(history.trainValue.values[-1], forecastHorizon)

def simpleESForecast(history, forecastHorizon, trainingWindow=None):
    trainData = (history.tail(trainingWindow) if trainingWindow else history).trainValue
    exponentialSmoothing = SimpleExpSmoothing(trainData)
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

def loadExperimentResults(experimentId, index=False):
    filePath = f'{EXPERIMENTS_RESULTS_PATH}/{experimentId}.csv'
    if os.path.isfile(filePath):
        data = pd.read_csv(filePath)
        if index:
            data = data.set_index(data['time'])
            return data
        return data

def loadExperimentsAndResults(datasetName):
    train, test = loadDataset(datasetName)
    experiments = getExperimentsTable()
    experiments = experiments[experiments.datasetName == datasetName]
    for i, experiment in experiments.iterrows():
        results = loadExperimentResults(experiment['id'], True)
        predictions = results['prediction']
        test[f"pred_{experiment['algorithmName']}_{experiment['hyperparams']}"] = predictions.to_list() + [None]*(len(test) - len(predictions))
    return experiments, test


def runExperiment(experiment):
    train, test = loadDataset(experiment['datasetName'])
    hyperparams = getHyperparams(experiment)
    algorithmName = experiment['algorithmName']
    fullseries = train.copy()[['value', 'time']]
    fullseries['trainValue'] = fullseries['value']
    fullseries['testValue'] = None
    oldResults = loadExperimentResults(experiment['id'])
    if oldResults is not None:
        fullseries = pd.concat([fullseries, oldResults])
    forecastHorizon = hyperparams['forecastHorizon']
    predict = predictFuncs[algorithmName]
    chunks = list(iterChunks(test, forecastHorizon))
    for i, chunk in enumerate(tqdm(chunks)):
        if len(fullseries) > len(train) + i*forecastHorizon: continue
        ypred = predict(fullseries, **hyperparams)
        newData = chunk.copy()[['testValue', 'time']]
        newData['trainValue'] = newData.testValue
        newData['prediction'] = ypred[:len(chunk)]
        fullseries = fullseries.append(newData)
        appendResults(experiment['id'], newData)
