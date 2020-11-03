import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report


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
#Cretion d'une novelle table contenant seulement les variables quantitatives
col = [ 1, 2, 4, 5, 6, 7, 12, 13]
print(col)

data_quanti = data1.iloc[: , col]
data_quanti.shape
data_quanti.info()


#Boxplot pour les variables quatitatives
plt.close()
plt.figure(1)
plt.figure(figsize=(10,12))
for i in np.arange(start=0,stop=data_quanti.shape[1]):
    plt.subplot(4, 2, i+1)
    plt.boxplot(data_quanti.iloc[:,i])
    plt.xlabel(data_quanti.columns[i])
plt.tight_layout()
plt.show() 

#Nous remarquons une valeur aberrante pour la variable term
#Faisons une histogramme de la variable term pour mieux identifier cette valeur
plt.hist(data_quanti.iloc[:,1])

data_quanti.iloc[:,1].describe()

#Détection d'outliers
data2=data1[(data1["term"]<500)] 
data2.shape


#----------------------------------------------------------
#                        Etape 2
#                   Analyse Univariée
#----------------------------------------------------------
#---2.1. Analyse univariee des variables quantitatives----##

# 2.1.1. Parametres des tendances centrales et indicateur de dispersion.
col = [ 1, 2, 4, 5, 6, 7, 12, 13]
data2.iloc[: , col].describe()

#2.1.2. Representation graphique des variables quantitatives

# Graphique: Distribution de la variable loanamt
plt.hist(data2.iloc[: , 1], bins=20, color='red')
plt.xlim(2, 980)
plt.ylim(0,900)
plt.xlabel('Montant du pret')
plt.ylabel('Effectif')
plt.title('Distribution de la variable loanamt')
plt.show()

# Graphique: Distribution de la variable term
plt.hist(data2.iloc[: , 2], bins=20, color='red')
plt.xlim(6, 480)
plt.ylim(0,1900)
plt.xlabel('Durée du pret en mois')
plt.ylabel('Effectif')
plt.title('Distribution de la variable term')
plt.show()


# Graphique: Distribution de la variable atotinc
plt.hist(data2.iloc[: , 4], bins=20, color='red')
plt.xlim(0, 81000)
plt.ylim(0,1200)
plt.xlabel('Revenu mensuel total')
plt.ylabel('Effectif')
plt.title('Distribution de la variable atotinc')
plt.show()

# Graphique: Distribution de la variable cototinc
plt.hist(data2.iloc[: , 5], bins=20, color='red')
plt.xlim(0, 41700)
plt.ylim(0,1200)
plt.xlabel('Coapp revenu mensuel total.')
plt.ylabel('Effectif')
plt.title('Distribution de la variable cototinc')
plt.show()

# Graphique: Distribution de la variable hrat
plt.hist(data2.iloc[: , 6], bins=70, color='red')
plt.xlim(0, 75)
plt.ylim(0,200)
plt.xlabel('Ratio fais logement/ revenu total.')
plt.ylabel('Effectif')
plt.title('Distribution de la variable hrat')
plt.show()

# Graphique: Distribution de la variable obrat
plt.hist(data2.iloc[: , 7], bins=70, color='red')
plt.xlim(0, 100)
plt.ylim(0,250)
plt.xlabel('Ratio autres dépenses/ revenu total.')
plt.ylabel('Effectif')
plt.title('Distribution de la variable obrat')
plt.show()


# Graphique: Distribution de la variable dep
plt.hist(data2.iloc[: , 12], bins=40, color='red')
plt.xlim(0, 9)
plt.ylim(0,1250)
plt.xlabel('Nombre de dépendants.')
plt.ylabel('Effectif')
plt.title('Distribution de la variable dep')
plt.show()

# Graphique: Distribution de la variable expr
plt.hist(data2.iloc[: , 13], bins=10, color='red')
plt.xlim(0, 10)
plt.ylim(0,1800)
plt.xlabel('Nombre d\'années d\'experience professionnel.')
plt.ylabel('Effectif')
plt.title('Distribution de la variable exr')
plt.show()

#---2.2. Analyse univariee des variables qualitatives----##

#Creation d'une novelle table contenant seulement les variables qualitatives
col1=[0,3,8,9,10,11,14,15,16,17,18,19]
data_quali = data2.iloc[:,col1]
data_quali.shape
data_quali.info()

#Renommer les modalités des variables binaires
data_quali["Approve_rec"] = data_quali.iloc[:,0].replace({1: "Oui", 0: "Non"})
data_quali["caution_rec"] = data_quali.iloc[:,1].replace({1: "Une Miise en Garde", 0: "Pas de Mise en Garde"})
data_quali["uni_rec"] = data_quali.iloc[:,3].replace({1: "Oui", 0: "Non"})
data_quali["married_rec"] = data_quali.iloc[:,5].replace({1: "Oui", 0: "Non"})
data_quali["self_rec"] = data_quali.iloc[:,6].replace({1: "Oui", 0: "Non"})
data_quali["delinq_rec"] = data_quali.iloc[:,8].replace({1: "Oui", 0: "Non"})
data_quali["mortperf_rec"] = data_quali.iloc[:,9].replace({1: "Oui", 0: "Non"})
data_quali["mortlat1_rec"] = data_quali.iloc[:,10].replace({1: "Oui", 0: "Non"})
data_quali["mortlat2_rec"] = data_quali.iloc[:,12].replace({1: "Oui", 0: "Non"})

data_quali.info()

# Tri à plat des variables
col3=[2,4,7,12,13,14,15,16,17,18,19,20]
for i in col3:
    tri_plat=data_quali.iloc[:,i].value_counts()
    print('Tri à plat de la variable',data_quali.columns[i],':','\n', tri_plat)
    print('\n')


#Pie des varibales qualitatives binaires
col3=[2,12,13,14,15,16,17,18,19,20]

j=1
plt.figure(figsize=(8,10))
for i in col3:
    plt.subplot(4, 3, j)
    j=j+1
    data_quali.iloc[:,i].value_counts().plot.pie(subplots=True, figsize = (4, 3) , autopct='%1.1f%%',startangle=90, colors = [ 'blue', 'yellow'])
    centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle) 
plt.tight_layout() 
plt.show()


# Bar plot de la variable race 
plt.bar(['White', 'Black', 'Hispan'], [1658, 193, 108], color=['red', 'blue','yellow'], width=0.8)
plt.ylabel('Effectif')
plt.title('Repartition des emprunteurs par race')
plt.show()


# Bar plot de la variable score
plt.bar([0,1,2,3,4,5,6,8,9], [174,970,610,91,101,4,7,1,1], color=[ 'blue'], width=0.8)
plt.ylabel('Effectif')
plt.xlabel('Score')
plt.title('Repartition des emprunteurs par score')
plt.show()
 
    
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

#
#FAIRE SELECTION DE VAR ICI
#

#Prediction
logreg = LogisticRegression()
modellogit=logreg.fit(os_data_X,os_data_y)
y_pred=logreg.predict(data_predX)

conf = confusion_matrix(data_predy, logreg.predict(data_predX))
cf = pd.DataFrame(conf, columns=[logreg.classes_])
cf.index = [ logreg.classes_]
cf

score = logreg.decision_function(data_predX)
df = {'score':score,'approve':data_predy,'pred':logreg.predict(data_predX)}
df=pd.DataFrame(data=df)

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

data_X,data_predX, data_y,  data_predy = train_test_split(data.drop(['approve'], axis=1),data["approve"], test_size=0.2, random_state=5, stratify=data["approve"])
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf.fit(data_X, data_y)

# Problème il me dit que j'ai des NaN

y_pred = clf.predict(data_predX)
accuracy = clf.score(data_predX, data_predy)
print(accuracy)

############################# sur le site de scikit learn

#from sklearn.ensemble import RandomForestClassifier
#X = [[0, 0], [1, 1]]
#Y = [0, 1]
#clf = RandomForestClassifier(n_estimators=10)
#clf = clf.fit(X, Y)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, data_X, data_y, cv=5) #j'ai mis le cv par défaut mais peut etre que je devrai changer
scores.mean()


clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, data_X, data_y, cv=5)
scores.mean()


clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, data_X, data_y, cv=5)
#scores.mean() > 0.999 changer la valeur


#----------------------------------------------------------
#                        Etape ???
#             Modelisation - AUTRES METHODE ?????
#---------------------------------------------------------- 



#----------------------------------------------------------
#                        Etape ???
#            Comparaison des performances de prédictions
#---------------------------------------------------------- 
