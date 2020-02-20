from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

names = ['v','i','label1']
dataframe = read_csv('file5.csv', header=None, names=names)

#feature_cols = ['v']

feature_cols = ['v','i']
X = dataframe[feature_cols]
y = dataframe.label1

#y=dataframe.label
# Independent variable binary for logistic regression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
kfold = KFold(n_splits=100, random_state=0)
model = Sequential()

#model = LinearRegression()
# applying deep neural network model
model.add(Dense(12, input_dim=2, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,y, epochs=5, batch_size=10)
_, accuracy = model.evaluate(X,y)

print('Accuracy: %.2f' % (accuracy*100))

