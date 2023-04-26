import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataTest = pd.read_csv("dataset/mnist_test.csv")
dataTrain = pd.read_csv("dataset/mnist_train.csv")

dataTest = np.array(dataTest)
dataTrain = np.array(dataTrain)

m, n = dataTrain.shape
np.random.shuffle(dataTrain)

dataTrain = dataTrain[0:m].T
dataTest = dataTest[0:m].T


yTest = dataTest[0]
xTest = dataTest[1:n]
yTrain = dataTrain[0]
xTrain = dataTrain[1:n]

