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
#Pour analyse univarié
#Ajouter histogramme ou boxplot
#Ajouter détection de outliers
for i in np.arange(start=0,stop=data.shape[1]):
    plt.plot(data.index.values,data.iloc[:,i],marker="o",ls=' ')
    plt.show()
#Ajouter Echantillonage 


# Pour l'analyse bivarié 
plt.figure(figsize=(10,10))
masque  =  np . tril ( data . corr ())
sns.heatmap(data.corr(),annot=True,vmin=-1, vmax=1,fmt='.2f',cmap= 'bwr' ,square=True,mask = masque)
plt.show()    

sns.pairplot(data)

#Ajouter analyse multivarié (ex: race*university*approve)
