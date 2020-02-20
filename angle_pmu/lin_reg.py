from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

names = ['v','i','label']
dataframe = read_csv('file.csv', header=None, names=names)
feature_cols = ['v']
#feature_cols = ['v','i']
X = dataframe[feature_cols]
y = dataframe.i
#y=dataframe.label
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
kfold = KFold(n_splits=100, random_state=0)
#model = LogisticRegression(solver='liblinear')
model = LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
#cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
#print(cnf_matrix)
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
sns.set_style('dark')
plot = sns.scatterplot(y_test,y_pred)
plot.set(xlabel='Given', ylabel='Prediction')
x_plot = np.linspace(0,5,100)
y_plot = x_plot
plt.plot(x_plot, y_plot, color='r')
plt.show()
