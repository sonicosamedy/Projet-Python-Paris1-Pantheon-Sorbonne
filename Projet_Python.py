import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import imblearn
import seaborn 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc


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
# Graphique: Distribution
b=[30,50,100,70,70,70,20,15]
plt.figure(figsize=(15,6))
for i in np.arange(start=0,stop=data_quanti.shape[1]):
    plt.subplot(2, 4, i+1)
    plt.hist(data_quanti.iloc[: , i], bins=b[i], color='darkseagreen')
    plt.xlabel(data_quanti.columns[i])
    plt.ylabel('Effectif')
    plt.title('Distribution de la variable ' + data_quanti.columns[i])
plt.tight_layout()
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
masque  =  np.tril(data2.corr())
sns.heatmap(data2.corr(),annot=True,annot_kws={"size": 7}, vmin=-1, vmax=1,fmt='.2f',cmap= 'bwr' ,square=True,mask = masque)
plt.show()      

#Attention c'est très long a éxécuter !
sns.pairplot(data2)

#OPTIMISER les histogrammes ci-dessous
#https://fxjollois.github.io/cours-2016-2017/analyse-donnees-massives-tp5.html
t1 = pd.crosstab(data.race, data.approve, normalize=True) #diagramme en barres
t1.plot.bar()

t2 = pd.crosstab(data.race, data.approve, normalize = "index") #pourcentages
t2

t3 = pd.crosstab(data.sex, data.approve, normalize=True)
t3.plot.bar()

t4 = pd.crosstab(data.sex, data.approve, normalize = "index")
t4

t5 = pd.crosstab(data.married, data.approve, normalize=True)
t5.plot.bar()

t6 = pd.crosstab(data.married, data.approve, normalize = "index")
t6

t7 = pd.crosstab(data.university, data.approve, normalize=True)
t7.plot.bar()

t8 = pd.crosstab(data.university, data.approve, normalize = "index")
t8

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

#Création des variables dummies
data2['exper'] = data2['exper'].astype(str)
data2 = pd.get_dummies(data2)
# Echantillonnage
data_X,test_X, data_y,  test_y = train_test_split(data2.drop(['approve'], axis=1),data2["approve"], test_size=0.2, random_state=5, stratify=data2["approve"])
#Rééquilibrage de la BDD d'entrainement
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


#  Selection des variables (Amélioration du critère BIC)

select_data=os_data_X.drop(['exper_9','dep','self','mortperf','mortlat2','mortlat1','hrat','loanamt','cototinc'], axis='columns')
select_test=test_X.drop(['exper_9','dep','self','mortperf','mortlat2','mortlat1','hrat','loanamt','cototinc'], axis='columns')

logit2 = sm.Logit(os_data_y,select_data.astype(float))
result2=logit2.fit()
print(result2.summary2())

#  Prédiction - Étude des performances de prédiction

logit2 = sm.Logit(os_data_y,select_data.astype(float))
result2=logit2.fit()
print(result2.summary2())

logreg = LogisticRegression()
modellogit=logreg.fit(select_data,os_data_y)
y_pred=logreg.predict(select_test)

conf = confusion_matrix(test_y, logreg.predict(select_test))
cf = pd.DataFrame(conf, columns=[logreg.classes_])
cf.index = [ logreg.classes_]
cf

fig, ax = plt.subplots() #Affichage a changer !
sns.heatmap(pd.DataFrame(conf), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

score = logreg.decision_function(select_test)
df = {'score':score,'approve':test_y,'pred':logreg.predict(select_test)}
df=pd.DataFrame(data=df)

ax = df[df['approve'] == 1]['score'].hist(bins=25, figsize=(6,3), label='1', alpha=0.5)
df[df['approve'] == 0]['score'].hist(bins=25, ax=ax, label='0', alpha=0.5)
ax.set_title("Distribution des scores pour les deux classes")
ax.plot([0, 0], [0, 55], 'g--', label="SEUIL ?")
ax.legend();


ax = seaborn.distplot(df[df['approve'] == 1]['score'], rug=True,bins=20, hist=True, label="1")
seaborn.distplot(df[df['approve'] == 0]['score'], rug=True, hist=True,bins=20, ax=ax, label="0")
ax.set_title("Distribution des scores pour les deux classes")
ax.legend()


#  Courbe ROC
pred_proba = logreg.predict_proba(select_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(test_y,  pred_proba)
auc = metrics.roc_auc_score(test_y, pred_proba)
plt.plot(fpr,tpr,label="Courbe ROC, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#  Courbe Rappel Précision
lr_precision, lr_recall, _ = precision_recall_curve(test_y, pred_proba)
lr_f1, lr_auc = f1_score(test_y, y_pred), auc(lr_recall, lr_precision)
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
plt.plot(lr_recall, lr_precision, label='Courbe Rappel-Précision, PR_auc='+str(lr_auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()




#----------------------------------------------------------
#                        Etape 7
#             Modelisation - Arbre de Décision
#----------------------------------------------------------  
#Nous cherchons à expliquer la variable approve avec les variable qualitative
col1=[0,3,8,9,10,11,14,15,16,17,18,19]
df = data2.iloc[:,col1]

#vérification de la version de scikit-learn
import sklearn
print(sklearn.__version__)

#Dimension du dataframe :
df.shape

#Affichages des premières lignes 
df.head()

#Affichage des informations sur le type de variables :
df.info()

#Vérifions la distribution absolue des approve
df.approve.value_counts()
#Distribution relative 
df.approve.value_counts(normalize=True)
'''
Ces informations sont importantes lorsque nous aurons à inspecter les résultats.
'''

'''
Je rencontre un soucis avec la variable de genre : sex 
qui est en string et la fonction a l'air de ne pas apprécier je vais la convertir
à l'aide d'un mapping 
'''
df['sex'] = df['sex'].map({'Male': 1,'Female': 0})
df.sex.value_counts()
'''
J'ai le même soucis avec race : 
    White = 1
    Black = 2
    Hispan = 3
'''
df['race'] = df['race'].map({'White': 1,'Black': 2, 'Hispan':3})
df.race.value_counts()

#Subdiviser les données en échantillons d'apprentissage et de test
#Nous nous allons prendre 70% pour le modèle de train et 30% pour le test
from sklearn.model_selection import train_test_split

dfTrain, dfTest = train_test_split(df,test_size=588,random_state=1,stratify=data2.approve)

#Vérification des dimensions 
dfTrain.shape #(1371, 12)
dfTest.shape #(588, 12)

#Vérification des distribution de approve
dfTrain.approve.value_counts(normalize=True)
dfTest.approve.value_counts(normalize=True)
'''
Les proportions sont respecté
'''

#instanciation de l'arbre
from sklearn.tree import DecisionTreeClassifier
arbreFirst = DecisionTreeClassifier(min_samples_split=150,min_samples_leaf=50)
#Je ne sais pas comment définir ces 2 paramètres 

#construction de l'arbre
'''
Je veux enlever la première colonne pour définir ma matrice X des variables prédictives 
et le vecteur Y la variable cible 
dfTrain.iloc[:,1:].columns
dfTrain.columns
'''

arbreFirst.fit(X = dfTrain.iloc[:,1:], y = dfTrain.approve) 

#affichage graphique de l'arbre - depuis sklearn 0.21
 #https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html#sklearn.tree.plot_tree 
from sklearn.tree import plot_tree
plot_tree(arbreFirst,feature_names = list(data2.columns[1:]),filled=True)

#affichage plus grand pour une meilleure lisibilité
plt.figure(figsize=(10,10))
plot_tree(arbreFirst,feature_names = list(data2.columns[1:]),filled=True) 
plt.show()





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
