from experiments import runExperiment, findOrCreateExpeiment, NAIVE, EXPONENTIAL_SMOOTHING, HOLT_WINTERS
from datasets import DATASET_NAMES

hyperparams = dict(
    forecastHorizon=1,
)

experimentBatch = [
    findOrCreateExpeiment(**args) for args in [
        dict(
            algorithmName=NAIVE,
            datasetName=DATASET_NAMES[0],
            hyperparams=dict(forecastHorizon=1)
        ),
        dict(
            algorithmName=EXPON ,
            datasetName=DATASET_NAMES[0],
            hyperparams=dict(forecastHorizon=1)
        ),
    ]
]

runExperiment(findOrCreateExpeiment(
    algorithmName=HOLT_WINTERS,
    datasetName=DATASET_NAMES[1],
    hyperparams=hyperparams
))
