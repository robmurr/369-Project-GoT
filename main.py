import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split




def bayes(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)

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
    print('Data loaded.')

    # Preprocessing
    # Numeric Columns
    numeric_cols = ['age', 'dateOfBirth', 'popularity', 'numDeadRelations']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())


    # Binary Columns
    binary_cols = ['isAliveMother', 'isAliveFather', 'isAliveHeir', 'isAliveSpouse', 'male',
                   'isMarried', 'isNoble', 'book1', 'book2', 'book3', 'book4', 'book5']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)


    # Categorical columns
    # categorical_cols = ['title', 'culture', 'mother', 'father', 'heir', 'house', 'spouse']
    # df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)

    df.drop(['name', 'S.No'], axis=1, inplace=True, errors='ignore')
    df.drop(['title', 'culture', 'mother', 'father', 'heir', 'house', 'spouse'], axis=1, inplace=True)

    x = df.drop(['isAlive'], axis=1)
    y = df['isAlive']

    x = x.to_numpy()
    y = y.to_numpy()

    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    bayes(x, y)







