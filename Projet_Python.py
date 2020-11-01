import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = pd.read_csv('/Loan_data.csv')

data.describe()
data.info()
data.isnull().sum()
data.head()
data=data.dropna() #Suppression de 26 lignes

#----------------------------------------------------------
#                        Etape 1
#   Visualisation des données pour la détection d'outlier
#
#----------------------------------------------------------
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

#Détection d'outliers
data=data[(data["term"]<500)] 

#----------------------------------------------------------
#                        Etape 2
#                   Analyse Univariée
#----------------------------------------------------------
col=[0,8,9,10,16,17]
j=1
plt.figure(figsize=(8,10))
for i in col:
    plt.subplot(2, 3, j)
    j=j+1
    data.iloc[:,i].value_counts().plot.pie(subplots=True, figsize = (3, 3) , autopct='%1.1f%%',startangle=90, colors = [ '#ae7181', '#a2bffe' ,'#a2cffe'])    
plt.tight_layout() 
plt.show()

#Si possible améliorer ce plot.pie !
data.iloc[:,15].value_counts().plot.pie(subplots=True, figsize = (6, 6) , autopct='%1.0f%%',startangle=90, colors = [ '#ae7181','#d58a94' ,'#c292a1', '#a2bffe' ,'#a2cffe','#658cbb','#3b5b92','#014182'])
 
    
#----------------------------------------------------------
#                        Etape 3
#                    Analyse Bivariée
#----------------------------------------------------------       
#Matrice de corrélation
plt.figure(figsize=(8,8))
masque  =  np.tril(data.corr())
sns.heatmap(data.corr(),annot=True,annot_kws={"size": 7}, vmin=-1, vmax=1,fmt='.2f',cmap= 'bwr' ,square=True,mask = masque)
plt.show()      

#Attention c'est très long a éxécuter !
sns.pairplot(data)

#OPTIMISER les histogrammes ci-dessous
t6 = pd.crosstab(data.sex, data.approve, normalize=True)
t6.plot.bar()
t7 = pd.crosstab(data.race, data.approve, normalize=True)
t7.plot.bar()
t8 = pd.crosstab(data.married, data.approve, normalize=True)
t8.plot.bar()

#----------------------------------------------------------
#                        Etape 4
#                   Analyse Multivariée
#----------------------------------------------------------  
#Ajouter analyse multivarié (ex: race*university*approve)

#Je veux afficher les stats descriptives sur les revenus  en fonctions de 
#si les prêt a été approuvé ou non
tab=[]
tab.append(data.loc[data['approve']==0,'atotinc'].describe())
tab.append(data.loc[data['approve']==1,'atotinc'].describe())
print(tab)



#----------------------------------------------------------
#                        Etape 5
#            Echantillonnage - Par Tirage stratifié
#----------------------------------------------------------

data_X,data_predX, data_y,  data_predy = train_test_split(data.drop(['approve'], axis=1),data["approve"], test_size=0.2, random_state=5, stratify=data["approve"])
#Création des variables dummies
data_X = pd.get_dummies(data_X)
data_predX=pd.get_dummies(data_predX)
#Rééquilibrage de la BDD
os = SMOTE(random_state=1)
os_data_X,os_data_y=os.fit_sample(data_X, data_y)
print("Longueur de la nouvelle BDD",len(os_data_X))
print(os_data_y.value_counts())

#----------------------------------------------------------
#                        Etape 6
#             Modelisation - Régression Logistique
#----------------------------------------------------------  

logit = sm.Logit(os_data_y, os_data_X.astype(float))
result=logit.fit()
print(result.summary2())
#FAIRE SELECTION DE VAR ICI

logit = LogisticRegression()
modellogit=logit.fit(data_X,data_y)
#Matrice de confusion
conf = confusion_matrix(data_predy, modellogit.predict(data_predX))
conf
cf = pd.DataFrame(conf, columns=[modellogit.classes_])
cf.index = [ modellogit.classes_]
cf
score = modellogit.decision_function(data_predX)
df = {'score':score,'approve':data_predy,'pred':modellogit.predict(data_predX)}
df=pd.DataFrame(data=df)
#Bcp de surapprentissage à corriger & OPTIMISER le graphique 
ax = df[df['approve'] == 1]['score'].hist(bins=25, figsize=(6,3), label='1', alpha=0.5)
df[df['approve'] == 0]['score'].hist(bins=25, ax=ax, label='0', alpha=0.5)
ax.set_title("Distribution des scores pour les deux classes")
ax.plot([0, 0], [0, 55], 'g--', label="SEUIL ?")
ax.legend();


#----------------------------------------------------------
#                        Etape 7
#             Modelisation - Arbre de Décision
#----------------------------------------------------------  





#----------------------------------------------------------
#                        Etape 8
#             Modelisation - Forêts Aléatoire
#---------------------------------------------------------- 




#----------------------------------------------------------
#                        Etape ???
#             Modelisation - AUTRES METHODE ?????
#---------------------------------------------------------- 



#----------------------------------------------------------
#                        Etape ???
#            Comparaison des performances de prédictions
#---------------------------------------------------------- 
