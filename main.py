import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    if len(sys.argv) == 2:
        data_filename = sys.argv[1]
    else:
        data_filename = 'game_of_thrones_train.csv'

    print('Loading data...')
    df = pd.read_csv(data_filename)

    x = df.drop(['isAlive'], axis=1, errors='ignore')
    y = df['isAlive']

    x = x.to_numpy()
    y = y.to_numpy()

    print('X:', x.shape)
    print('X:', y.shape)







