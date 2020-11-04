from experiments import runExperiment, findOrCreateExpeiment, NAIVE, EXPONENTIAL_SMOOTHING, HOLT_WINTERS, ARIMA_FORECAST
from datasets import spxLoader, electricDemandLoader, seriaMaLoader
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from utils import fourierTransform

spxTrain, _ = spxLoader.load()
electricDemandTrain, _ = electricDemandLoader.load()
seriaMaTrain, _ = seriaMaLoader.load()

def getMainPeriods(data, numberOfPeriods):
    transform = fourierTransform(data)
    transform = transform[np.isfinite(transform.period)]
    transform['period'] = transform.period.apply(round)
    transform = transform[transform.period <= len(data)/2]
    periods = transform.sort_values(by='magnitude', ascending=False).head(numberOfPeriods).period.to_list()
    return periods

spxPeriods = getMainPeriods(spxTrain[spxLoader.valueColumn], 3)
electricDemandPeriods = getMainPeriods(electricDemandTrain[electricDemandLoader.valueColumn], 3)
seriaMaPeriods = getMainPeriods(seriaMaTrain[seriaMaLoader.valueColumn], 3)
print(spxPeriods, electricDemandPeriods, seriaMaPeriods)

spxForecastHorizon = 1 # 1 dia a frente
electricDemandForecastHorizon = 24 # 1 dia a frente
seriaMaForecastHorizon = 7 # 1 semana a frente

experimentBatch = [
    findOrCreateExpeiment(**args) for args in [
        # SPX
        dict(
            algorithmName=NAIVE,
            datasetName=spxLoader.datasetName,
            hyperparams=dict(
                forecastHorizon=spxForecastHorizon
            )
        ),
        dict(
            algorithmName=EXPONENTIAL_SMOOTHING,
            datasetName=spxLoader.datasetName,
            hyperparams=dict(
                forecastHorizon=spxForecastHorizon
            )
        ),
        *[
            dict(
                algorithmName=HOLT_WINTERS,
                datasetName=spxLoader.datasetName,
                hyperparams=dict(
                    forecastHorizon=spxForecastHorizon,
                    seasonal_periods=period,
                    trend='add',
                    seasonal='add'
                )
            ) for period in spxPeriods
        ],
        dict(
            algorithmName=ARIMA_FORECAST,
            datasetName=spxLoader.datasetName,
            hyperparams=dict(
                forecastHorizon=spxForecastHorizon,
                trainingWindow=600,
                p=2, d=1, q=1
            )
        ),

        # CECOVEL
        dict(
            algorithmName=NAIVE,
            datasetName=electricDemandLoader.datasetName,
            hyperparams=dict(
                forecastHorizon=electricDemandForecastHorizon
            )
        ),
        dict(
            algorithmName=EXPONENTIAL_SMOOTHING,
            datasetName=electricDemandLoader.datasetName,
            hyperparams=dict(
                forecastHorizon=electricDemandForecastHorizon
            )
        ),
        *[
            dict(
                algorithmName=HOLT_WINTERS,
                datasetName=electricDemandLoader.datasetName,
                hyperparams=dict(
                    forecastHorizon=electricDemandForecastHorizon,
                    seasonal_periods=period,
                    trend='add',
                    seasonal='add'
                )
            ) for period in electricDemandPeriods
        ],
        dict(
            algorithmName=ARIMA_FORECAST,
            datasetName=electricDemandLoader.datasetName,
            hyperparams=dict(
                forecastHorizon=electricDemandForecastHorizon,
                trainingWindow=600,
                p=2, d=1, q=1
            )
        ),

        # SERIA MA
        dict(
            algorithmName=NAIVE,
            datasetName=seriaMaLoader.datasetName,
            hyperparams=dict(
                forecastHorizon=seriaMaForecastHorizon
            )
        ),
        dict(
            algorithmName=EXPONENTIAL_SMOOTHING,
            datasetName=seriaMaLoader.datasetName,
            hyperparams=dict(
                forecastHorizon=seriaMaForecastHorizon
            )
        ),
        *[
            dict(
                algorithmName=HOLT_WINTERS,
                datasetName=seriaMaLoader.datasetName,
                hyperparams=dict(
                    forecastHorizon=seriaMaForecastHorizon,
                    seasonal_periods=period,
                    trend='add',
                    seasonal='add'
                )
            ) for period in seriaMaPeriods
        ],
        dict(
            algorithmName=ARIMA_FORECAST,
            datasetName=seriaMaLoader.datasetName,
            hyperparams=dict(
                forecastHorizon=seriaMaForecastHorizon,
                trainingWindow=600,
                p=2, d=1, q=1
            )
        ),
    ]
]

print(len(experimentBatch), 'experiments to run')

Parallel(-1)(delayed(runExperiment)(experiment) for experiment in tqdm(experimentBatch))
