import sys
import numpy as np





## Examples taken from assignments

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataMat.append(curLine)
    return np.array(dataMat)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        data_filename = sys.argv[1]
    else:
        data_filename = 'game_of_thrones_train.csv'

    print('Loading data...')
    dataMat = loadDataSet(data_filename)





