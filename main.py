import importlib
import subprocess
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score, classification_report,
)




def bayes(X, y):

    # Split the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)

    # Build a Gaussian Classifier
    model = GaussianNB()

    # Model training
    model.fit(X_train, y_train)

    # Predict Output
    y_pred = model.predict(X_test)

    # print("Actual Value:", y_test)
    # print("Predicted Value:", y_pred)

    # Calculate values
    accuracy = accuracy_score(y_pred, y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:", conf_matrix)
    print("Classification Report:", class_report)

    # Create matplot figure
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['Deceased', 'Alive'], yticklabels=['Deceased', 'Alive'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("ConfusionMatrix.png")
    plt.show()
    plt.close()


def install_requirements():
    requirements = [
        'numpy==1.25.2',
        'pandas==2.2.3',
        'seaborn==0.13.2',
        'matplotlib==3.10.0',
        'scikit-learn==1.6.1'
    ]

    for req in requirements:
        package = req.split('==')[0]
        try:
            importlib.import_module(package)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            except subprocess.CalledProcessError as e:
                sys.exit(1)


if __name__ == '__main__':
    # Install requirements
    install_requirements()

    # Read arguements
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







