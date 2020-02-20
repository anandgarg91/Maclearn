import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
data = pandas.read_csv('file.csv')
scatter_matrix(data)
plt.show()
