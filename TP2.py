#!/usr/bin/env python
# coding: utf-8

# # TP2 analyse et apprentissage supervisé
# 

# In[2]:


import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

#from utils_titanic import plot_confusion_matrix


# # PARTIE I :

# ## Exercice 1:

#     1. Chargement des données et calculs des valeurs manquantes

# In[3]:


import os 
os.getcwd()


# In[4]:



    PassengerId : numéro du passager
    Survived : 1 si le passager a survécu, sinon 0
    Pclass : La classe dans laquelle le passager a voyagé (1, 2 ou 3)
    Name : Le titre et le nom
    Sex : Homme ou femme
    Age :
    SibSp : Nombre de conjoints et frères/sœurs à bord
    Parch : Nombre de parents/enfants à bord
    Ticket : référence du ticket
    Fare : prix payé
    Cabin : numéro de cabine (pour ceux qui en ont)
    Embarked : Port de départ (S = Southampton, C = Cherbourg, Q = Queenstown)


# In[5]:


import pandas as pd
train = pd.read_csv('./train.csv',sep=',')
test = pd.read_csv('./test.csv',sep=',')

print("Taille de l'échantillon d'apprentissage  " + str(train.shape) + "\n" + 
      "Taille de l'échantillon de test  " + str(test.shape) + "\n\n")

train.sample(1)


# In[6]:


# missing values
pd.set_option('display.max_rows', 10)
train
#train.isnull().sum()


# In[7]:


train.head()


# In[8]:


for i in train.columns:
    print(i, round((train.shape[0] - 
                    train[i].count())/train.shape[0]*100,3))


# In[9]:


get_ipython().system('pip -q install pandas_profiling')
import pandas_profiling
profile = train.profile_report(title='Pandas Profiling Report').


# ## Exercice 2

# In[10]:


# Valeur manquantes de l'âge
print("Nombre de passagers avec un âge à null -->" + "\n" +
      str(train.Age.isnull().sum()) +  ' passagers \n\n\n')

train.loc[train.Age.isnull()].sample(5)


# In[11]:


# Croisement avec la Class
train.groupby('Pclass').agg(
     mean=pd.NamedAgg(column='Age', aggfunc='mean'),
        min=pd.NamedAgg(column='Age', aggfunc='min'),
        max=pd.NamedAgg(column='Age', aggfunc='max'))\
    .sort_values('mean',ascending =False)


# In[12]:


train[['PassengerId','Pclass']]


# In[13]:


# Analyse croisée entre l'age 
fig, ax = plt.subplots(1,1,figsize=(8,5))
train.loc[train.Pclass == 1, 'Age'].hist(alpha=0.6,ax=ax)
train.loc[train.Pclass == 2, 'Age'].hist(alpha=0.4,ax=ax)
train.loc[train.Pclass == 3, 'Age'].hist(alpha=0.2,ax=ax)
ax.legend(['1','2','3'])

print("Pclass = 1: age moyen ->",round(train.loc[train.Pclass == 1, 'Age'].mean(),2))
print("Pclass = 2: age moyen ->",round(train.loc[train.Pclass == 2, 'Age'].mean(),2))
print("Pclass = 3: age moyen ->",round(train.loc[train.Pclass == 3, 'Age'].mean(),2))


# In[14]:


# choix pour l'Age : analyse croisée des valeurs manquantes pour la var Age (avec Pclass)
train.loc[train.Pclass == 1, 'Age'] =     train[train.Pclass == 1].Age.fillna(train.loc[train.Pclass == 1, 'Age'].mean())

train.loc[train.Pclass == 2, 'Age'] =             train[train.Pclass == 2].Age.fillna(train.loc[train.Pclass == 2, 'Age'].mean())

train.loc[train.Pclass == 3, 'Age'] =             train[train.Pclass == 3].Age.fillna(train.loc[train.Pclass == 3, 'Age'].mean())


# In[15]:


train[train.Embarked.isnull()]


# In[16]:


train[train.Embarked.isnull()]

# Une recherche sur internet des noms des passagers nous donne l'info: 'S'
# -> https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html
# -> https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html
train.Embarked.fillna('S', inplace=True)


# In[17]:


# example de scatter plot afin de détecter d'éventuelles valeurs aberrantes/extrêmes

fig, ax = plt.subplots(1,1, figsize=(10,10))
pd.plotting.scatter_matrix(train[['Age', 'SibSp', 'Parch', 'Fare']],                            diagonal='hist',
                           alpha=0.8, ax=ax)


# In[18]:


train.Fare.sort_values(ascending=False).head(5)


# In[19]:


fig, ax = plt.subplots(1,2,figsize=(10,5))
# Variable Fare possédent 3 valeurs aberrantes a priori : 
#nous faisons le choix ainsi de les remplacer par le max

#train[train.Pclass==1].Fare.hist(ax=ax[0])
train.Fare.hist(ax=ax[0],)
ax[0].set_title('Before outliers replaced')

# remplacement par le max de Pclass
maxi = train[(train.Fare<500)].Fare.max()
#maxi = train[(train.Pclass==1)&(train.Fare<500)].Fare.max()
train.Fare = train.Fare.replace(512.3292, maxi)

#train[train.Pclass==1].Fare.hist(ax=ax[1])
train.Fare.hist(ax=ax[1])
ax[1].set_title('After outliers replaced')


# In[20]:


train.isnull().sum()


# # PARTIE II : Modélisation

# ## Exercice 3 :

# In[21]:


# selection de variable 
categorical_cols = ['Sex', 'Embarked','Pclass']
numeric_cols = ['Age', 'SibSp', 'Parch', 'Fare']
my_cols = categorical_cols + numeric_cols
target = 'Survived'

X, y = train[my_cols], train[target]


# In[22]:


X.corr()


# In[23]:


# Corrélation
import seaborn as sns
def plot_correlation_map( df ):
    sns.reset_orig()
    corr = df.corr()
    fig , ax = plt.subplots(1,1,figsize =( 10,10 ) )
    cmap = sns.diverging_palette( 200 , 10 , as_cmap = True )
    fig = sns.heatmap(
            corr, 
            cmap = cmap,
            square=True, 
            cbar_kws={ 'shrink' : 0.9 }, 
            ax=ax, 
            annot = True, 
            annot_kws = { 'fontsize' : 10 }
        )
plot_correlation_map(train[my_cols])


# In[24]:


# one Hot Encoding / get_dummies
X = pd.get_dummies(X, columns=categorical_cols)
X.head()


# In[25]:


# split train, valid
X_train, X_valid, y_train, y_valid =         train_test_split(X, y, test_size=0.2, random_state=4)

my_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Embarked_C',
       'Embarked_Q', 'Embarked_S', 'Pclass_1', 'Pclass_2', 'Pclass_3']

X_train = X_train[my_cols]
X_valid = X_valid[my_cols]


# In[26]:


X_valid.head()


# In[66]:


# Etude d'un hyperparametre de l'arbre de decision (profondeur de l'arbre)
res = list()
res_valid = list()
for i in range(1,30):
    model = DecisionTreeClassifier(max_depth=i)
    model.fit(X_train, y_train)
    res.append(model.score(X_train, y_train))
    res_valid.append(model.score(X_valid, y_valid))


# In[30]:


# plot les performances (accuracy score) for different depth of decision tree
fig, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(res)
ax.plot(res_valid)
ax.set_ylim(0.7,1)
ax.legend(['train','valid'])
ax.set_title('max_depth evolution on train and valid dataset')
plt.grid(True)


# In[31]:


# On choisit une profondeur de 6
model = DecisionTreeClassifier(max_depth=6)
model.fit(X_train, y_train)

# Prediction 
y_pred_train = model.predict_proba(X_train)
y_pred_valid = model.predict_proba(X_valid)

# Train probability prediction
y_pred_train = y_pred_train[:, 1]
y_pred_valid = y_pred_valid[:, 1]


# In[32]:


# ROC - Precision/recall curves

fig, ax = plt.subplots(1,2,figsize=(15,5))

#valid
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_valid)
auc =  round(roc_auc_score(y_valid, y_pred_valid),2)

#train
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_valid)
fpr_train, tpr_train, thresholds = roc_curve(y_train, y_pred_train)

ax[0].plot(fpr, tpr, lw=1, alpha=0.8)
ax[0].plot(fpr_train, tpr_train, lw=1, alpha=0.8)
ax[0].grid(True)
ax[0].legend(['valid (AUC='+str(auc)+')', 'train'])
ax[0].set_title('ROC curve')

# Precision / Recall performance
precision, recall, _ = precision_recall_curve(y_valid, y_pred_valid)
precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_train)

ax[1].step(recall, precision, alpha=0.8,
         where='post')
ax[1].step(recall_train, precision_train, alpha=0.8,
         where='post')
ax[1].set_xlabel('Recall')
ax[1].set_ylabel('Precision')
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlim([0.0, 1.0])
ax[1].grid(True)
ax[1].legend(['valid','train'])
ax[1].set_title('Precision / Recall curve')


# In[70]:


# choix d'un seuil et affichage de la matrice de confusion
y_pred_t = [1 if i > 0.95 else 0 for i in y_pred_valid]
confusion_matrix(y_valid, y_pred_t)
confusion_matrix(y_valid, y_pred_t).ravel()
# (tn, fp, fn, tp)


# ## Exercice 5: comparaison avec un RandomForest

# In[40]:


# utilisation de la fonction GridSearch permettant de tester differents hyperparametres à l'aide de la validation croisée
# warning: temps d'execution important 

param_grid = {'bootstrap': [True, False],
 'max_depth': [5,10],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2 ],
 'min_samples_split': [2, 5 ],
 'n_estimators': [5,10, 100]}

gs = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
gs.fit(X_train, y_train)
gs.best_params_


# In[21]:


# choisit le résultat de gridsearch
rf = RandomForestClassifier(bootstrap= True,
 max_depth= 50,
 max_features= 'auto',
 min_samples_leaf= 4,
 min_samples_split= 2,
 n_estimators= 30)
rf.fit(X_train, y_train)

# Prediction 
y_pred_train_rf = rf.predict_proba(X_train)
y_pred_valid_rf = rf.predict_proba(X_valid)

# Train probability prediction
y_pred_train_rf = y_pred_train_rf[:, 1]
y_pred_valid_rf = y_pred_valid_rf[:, 1]


#     1. Etude des performances du RandomForestClassifier 

# In[22]:


# ROC - Precision/recall curves

fig, ax = plt.subplots(1,2,figsize=(15,5))

#valid
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_valid_rf)
auc =  round(roc_auc_score(y_valid, y_pred_valid_rf),2)

#train
fpr_train, tpr_train, thresholds = roc_curve(y_train, y_pred_train_rf)

ax[0].plot(fpr, tpr, lw=1, alpha=0.8)
ax[0].plot(fpr_train, tpr_train, lw=1, alpha=0.8)
ax[0].grid(True)
ax[0].legend(['valid (AUC='+str(auc)+')', 'train'])
ax[0].set_title('ROC curve')

# Precision / Recall performance

precision_rf, recall_rf, _ = precision_recall_curve(y_valid, y_pred_valid_rf)
precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_train_rf)

ax[1].step(recall_rf, precision_rf, alpha=0.8,
         where='post')
ax[1].step(recall_train, precision_train, alpha=0.8,
         where='post')
ax[1].set_xlabel('Recall')
ax[1].set_ylabel('Precision')
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlim([0.0, 1.0])
ax[1].grid(True)
ax[1].legend(['valid','train'])
ax[1].set_title('Precision / Recall curve')


#     2. Comparaison des deux modèles : DecisionTreeClassifier et RandomForestClassifier

# In[23]:


fig, ax = plt.subplots(1,1,figsize=(10,5))

# Precision / Recall performance
auc_rf =  round(roc_auc_score(y_valid, y_pred_valid_rf),2)
auc_dt =  round(roc_auc_score(y_valid, y_pred_valid),2)

ax.step(recall, precision, alpha=0.8, where='post')
ax.step(recall_rf, precision_rf, alpha=0.8, where='post')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.0])
ax.grid(True)
ax.legend(['decision tree','random forest'])
ax.set_title('Precision / Recall curve')


# In[50]:


# Kmeans sur l'âge et le prix 
# Attention il faut normalement normaliser les variables car non homogénéité des dimensions
from sklearn.cluster import KMeans, DBSCAN, OPTICS

a=[]
nbK = 20
for i in range(2,nbK):
    kmeans = KMeans(n_clusters=i) 
    kmeans.fit(train[["Age","Fare"]]) 
    a.append(kmeans.inertia_)
plt.scatter(range(2,nbK),a)
plt.xlabel('Valeurs de K')
plt.ylabel('Inertie intra classe')
plt.grid(True)


# In[ ]:


kmeans = KMeans(n_clusters=5, n_init=1, init='random').fit(train[["Age","Fare"]])
import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [4,1,3,6]
size = [100,500,100,500]

plt.scatter(x, y, s=size, c='coral')


# In[64]:


kmeans = KMeans(n_clusters=5, n_init=1, init='random').fit(train[["Age","Fare"]])
fig, ax = plt.subplots(1,1,figsize=(10,5))

plt.scatter(train["Age"], train["Fare"],c=kmeans.labels_)

ax.set_title('Age graph')
plt.show()


# In[ ]:




