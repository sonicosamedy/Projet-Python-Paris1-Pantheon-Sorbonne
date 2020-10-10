import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/Loan_data.csv')

data.describe()
data.info()

data.isnull().sum()

data=data.dropna()
data.head()

for i in np.arange(start=0,stop=data.shape[1]):
    plt.plot(data.index.values,data.iloc[:,i],marker="o",ls=' ')
    plt.show()
