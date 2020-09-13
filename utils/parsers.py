import pandas as pd
import numpy as np
import dateutil.parser

MONTH_DICT = {
    'jan': 'jan',
    'fev': 'feb',
    'mar': 'mar',
    'abr': 'apr',
    'mai': 'may',
    'jun': 'jun',
    'jul': 'jul',
    'ago': 'aug',
    'set': 'sep',
    'out': 'oct',
    'nov': 'nov',
    'dez': 'dec'
}

def isOnlyDate(date):
    return date.hour == 0 and \
           date.minute == 0 and \
           date.second == 0 and \
           date.microsecond == 0

def parseDataset(dataset: pd.DataFrame, timeColumn: str) -> pd.DataFrame:
    dataset = dataset.set_index(
        dataset[timeColumn].apply(dateutil.parser.parse)
    )
    dateOnly = all(isOnlyDate(date) for date in dataset.index)
    if dateOnly: dataset = dataset.set_index(dataset.index.date)
    return dataset

def decomposeDate(dataset: pd.DataFrame, timeColumn: str = 'time', inplace=False):
    if not inplace: dataset = dataset.copy()
    dataset[['year', 'month', 'day', 'weekday', 'hour']] = dataset[timeColumn].apply(
        lambda date: pd.Series([
            date.year,
            date.month,
            date.day,
            date.weekday(),
            date.hour if hasattr(date, 'hour') else -1
        ])
    )
    return dataset

def translateMonthFromDate(datestr: str):
    for ptMonth, engMonth in MONTH_DICT.items():
        if ptMonth in datestr.lower():
            return datestr.lower().replace(ptMonth, engMonth)
    return datestr
