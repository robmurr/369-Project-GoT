import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split




def bayes(X, y, test_size, random_state):

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size, random_state)

    # Build a Gaussian Classifier
    model = GaussianNB()

    # Model training
    model.fit(X_train, y_train)

    # Predict Output
    predicted = model.predict([X_test[6]])

    print("Actual Value:", y_test[6])
    print("Predicted Value:", predicted[0])

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

    bayes(x, y, test_size=.33, random_state=125)







