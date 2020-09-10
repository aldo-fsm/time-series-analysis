import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
data_path = './datasets_55485_106148_spx.csv'
data = pd.read_csv(data_path)
print(data.shape)

data['datetime'] = data.date.apply(lambda date_str: datetime.strptime(date_str, '%d-%b-%y'))
data = data.sort_values(by='datetime', ascending=True).drop(columns=['datetime'])
train, test = train_test_split(data, shuffle=False, test_size=0.2)
print(
    train.shape, test.shape,
)
train.to_csv(data_path.replace('.csv', '_train.csv'), index=False)
test.to_csv(data_path.replace('.csv', '_test.csv'), index=False)
