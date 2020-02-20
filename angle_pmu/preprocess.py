# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
dataframe = pandas.read_csv('file.csv')
array = dataframe.values
# separate array into input and output components
X = array[:,0:1]
Y = array[:,1]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
