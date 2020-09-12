from utils.parsers import parseDataset, translateMonthFromDate
import pandas as pd

DATASETS_PATH = 'datasets'

def loadDataFrame(datasetName, datasetFormat, subsetName):
    filePath = f'{DATASETS_PATH}/{datasetName}_{subsetName}.{datasetFormat}'
    if datasetFormat.lower() == 'csv':
        return pd.read_csv(filePath)
    elif datasetFormat.lower() in ['xls', 'xlsx']:
        return pd.read_excel(filePath)


class DatasetLoader:
    def __init__(self, datasetName, datasetFormat, timeColumn, valueColumn, customTimeColumnFormatter=None):
        self.datasetName = datasetName
        self.datasetFormat = datasetFormat
        self.timeColumn = timeColumn
        self.valueColumn = valueColumn
        self.customTimeColumnFormatter = customTimeColumnFormatter if customTimeColumnFormatter else lambda timestr: timestr

    def load(self):
        train = loadDataFrame(self.datasetName, self.datasetFormat, 'train')
        train[self.timeColumn] = train[self.timeColumn].apply(self.customTimeColumnFormatter)
        train = parseDataset(train, self.timeColumn)

        test = loadDataFrame(self.datasetName, self.datasetFormat, 'test')
        test[self.timeColumn] = test[self.timeColumn].apply(self.customTimeColumnFormatter)
        test = parseDataset(test, self.timeColumn)

        return train, test

spxLoader = DatasetLoader(
    datasetName = 'datasets_55485_106148_spx',
    datasetFormat = 'csv',
    timeColumn = 'date',
    valueColumn = 'close'
)

electricDemandLoader = DatasetLoader(
    datasetName = 'ElectricDemandForecasting-DL-master_data_CECOVEL',
    datasetFormat = 'csv',
    timeColumn = 'timestamp',
    valueColumn = 'value'
)

electricDemandHourlyLoader = DatasetLoader(
    datasetName = 'ElectricDemandForecasting-DL-master_data_hourly_20140102_20191101',
    datasetFormat = 'csv',
    timeColumn = 'datetime',
    valueColumn = 'value'
)

seriaMaLoader = DatasetLoader(
    datasetName = 'SERIA_MA',
    datasetFormat = 'csv',
    timeColumn = 'date',
    valueColumn = 'value',
    customTimeColumnFormatter=translateMonthFromDate
)

LOADERS = [
    spxLoader,
    electricDemandLoader,
    electricDemandHourlyLoader,
    seriaMaLoader
]

DATASET_NAMES = [loader.datasetName for loader in LOADERS]

def getLoaderByName(name) -> DatasetLoader:
    for loader in LOADERS:
        if name == loader.datasetName: return loader
    raise Exception(f'Dataset {name} not found')
