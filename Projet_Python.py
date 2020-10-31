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
plt.close()
plt.figure(1)
plt.figure(figsize=(10,12))
for i in np.arange(start=0,stop=data.shape[1]):
    plt.subplot(7, 3, i+1)
    plt.plot(data.index.values,data.iloc[:,i],marker="o",ls=' ',alpha=0.7,color='#0f6f80')
    plt.xlabel('Indices d\'observation ')
    plt.ylabel(data.columns[i])
plt.tight_layout() 
plt.show()

t1 = pd.crosstab(data.sex, "freq")
t1.plot.pie(subplots=True, figsize = (6, 6))
t2 = pd.crosstab(data.race, "freq")
t2.plot.pie(subplots=True, figsize = (6, 6))
t3 = pd.crosstab(data.university, "freq")
t3.plot.pie(subplots=True, figsize = (6, 6))
t4 = pd.crosstab(data.married, "freq")
t4.plot.pie(subplots=True, figsize = (6, 6))
t5 = pd.crosstab(data.self, "freq")
t5.plot.pie(subplots=True, figsize = (6, 6))

t6 = pd.crosstab(data.sex, data.approve, normalize=True)
t6.plot.bar()
t7 = pd.crosstab(data.race, data.approve, normalize=True)
t7.plot.bar()
t8 = pd.crosstab(data.married, data.approve, normalize=True)
t8.plot.bar()

#Ajouter Echantillonage 
data_prd=data.sample(frac=0.2)#prendre 20% de la BDD
data=data.iloc[data.index.difference(data_prd.index),]

# Pour l'analyse bivarié 
plt.figure(figsize=(10,10))
masque  =  np . tril ( data . corr ())
sns.heatmap(data.corr(),annot=True,vmin=-1, vmax=1,fmt='.2f',cmap= 'bwr' ,square=True,mask = masque)
plt.show()    

sns.pairplot(data)

#Ajouter analyse multivarié (ex: race*university*approve)

#Je veux afficher les stats descriptives sur les revenus  en fonctions de 
#si les prêt a été approuvé ou non
tab=[]
tab.append(data.loc[data['approve']==0,'atotinc'].describe())
tab.append(data.loc[data['approve']==1,'atotinc'].describe())
print(tab)

