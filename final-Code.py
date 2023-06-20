#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import array
import pandas as pd 
from matplotlib import pyplot as plt
import plotly.express as px
import warnings
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns 
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from scipy.stats import randint as sp_randint
from matplotlib import colors
from sklearn.svm import SVC 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from plotly import tools
from plotly.subplots import make_subplots
from plotly.offline import iplot
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
def s():
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=.24).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']

    param_dist_svc = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['poly', 'rbf', 'linear', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'coef0': [-1, 0, 1]
    }
    model_svc = SVC()
    random_search_svc = RandomizedSearchCV(estimator=model_svc, param_distributions=param_dist_svc, scoring='roc_auc')
    random_search_svc.fit(X_train_labeled, y_train_labeled)
    model_svc_best = SVC(**random_search_svc.best_params_, probability=True)
    model_svc_best.fit(X_train_labeled, y_train_labeled)
    y_pred = model_svc_best.predict(X_test)

    a1=roc_auc_score(y_test, y_pred)
    b1=precision_score(y_test, y_pred)
    c1=f1_score(y_test, y_pred)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=.24).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']

    param_dist_logreg = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    }
    model_logreg = LogisticRegression()
    random_search_logreg = RandomizedSearchCV(estimator=model_logreg, param_distributions=param_dist_logreg, scoring='roc_auc')
    random_search_logreg.fit(X_train_labeled, y_train_labeled)
    model_logreg_best = LogisticRegression(**random_search_logreg.best_params_)
    model_logreg_best.fit(X_train_labeled, y_train_labeled)
    y_pred = model_logreg_best.predict(X_test)

    a2=roc_auc_score(y_test, y_pred)
    b2=precision_score(y_test, y_pred)
    c2=f1_score(y_test, y_pred)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=.24).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']


    param_dist =  {
    'n_estimators': [50, 100, 200, 300],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']}

    model = RandomForestClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist , scoring='roc_auc')
    random_search.fit(X_train_labeled, y_train_labeled)
    modelrfs = RandomForestClassifier(**random_search.best_params_)
    modelrfs.fit(X_train_labeled, y_train_labeled)
    y_pred = modelrfs.predict(X_test)

    a3=roc_auc_score(y_test, y_pred)
    b3=precision_score(y_test, y_pred)
    c3=f1_score(y_test, y_pred)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=.24).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    
    param_dist =  {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-10, 1e-11, 1e-12, 1e-4]}

    model = GaussianNB()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist , scoring='roc_auc')
    random_search.fit(X_train_labeled, y_train_labeled)
    modelnb = GaussianNB(**random_search.best_params_)
    modelnb.fit(X_train_labeled, y_train_labeled)
    y_pred = modelnb.predict(X_test)

    a4=roc_auc_score(y_test, y_pred)
    b4=precision_score(y_test, y_pred)
    c4=f1_score(y_test, y_pred)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=.24).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    
    param_dist = {'n_neighbors': [5, 7, 9, 10], 'leaf_size': [20,30,50],
             'metric': ['euclidean', 'manhattan' , 'minkowski']}

    model = KNeighborsClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist , scoring='roc_auc')
    random_search.fit(X_train_labeled, y_train_labeled)
    modelkn = KNeighborsClassifier(**random_search.best_params_)
    modelkn.fit(X_train_labeled, y_train_labeled)
    y_pred = modelkn.predict(X_test)

    a5=roc_auc_score(y_test, y_pred)
    b5=precision_score(y_test, y_pred)
    c5=f1_score(y_test, y_pred)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=.24).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']


    param_dist = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [1 ,2, 3, 4, 5, 6, 7, 8],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8]}

    model = DecisionTreeClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist , scoring='roc_auc')
    random_search.fit(X_train_labeled, y_train_labeled)
    modelDT = DecisionTreeClassifier(**random_search.best_params_)
    modelDT.fit(X_train_labeled, y_train_labeled)
    y_pred = modelDT.predict(X_test)

    a6=roc_auc_score(y_test, y_pred)
    b6=precision_score(y_test, y_pred)
    c6=f1_score(y_test, y_pred)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=.24).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']

    param_dist = {
    'n_estimators': sp_randint(50,100, 200),
    'learning_rate': [0.1, 0.01, 0.001 , 1],
    'max_depth': sp_randint(2, 10),
    'subsample': [0.5, 2, 1.0]}
    
    model = GradientBoostingClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist,scoring='recall',cv=3)
    random_search.fit(X_train_labeled, y_train_labeled)
    df_train_labeled=df_train[df_train['New_Response']!=-1]
    modelgb = GradientBoostingClassifier(**random_search.best_params_)
    modelgb.fit(X_train_labeled, y_train_labeled)
    y_pred = modelgb.predict(X_test)

    a7=roc_auc_score(y_test, y_pred)
    b7=precision_score(y_test, y_pred)
    c7=f1_score(y_test, y_pred)
    
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=.24).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    
    param_dist = {
    "alpha": [0.0001, 0.001, 0.01, 0.00001],
    "learning_rate": ["constant", "adaptive"],
    "hidden_layer_sizes": [(100,),(50,), (50, 50), (200,), (100, 100), (200, 100)]}

    model = MLPClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist , scoring='roc_auc')
    random_search.fit(X_train_labeled, y_train_labeled)
    modelpe = MLPClassifier(**random_search.best_params_)
    modelpe.fit(X_train_labeled, y_train_labeled)
    y_pred = modelpe.predict(X_test)

    a8=roc_auc_score(y_test, y_pred)
    b8=precision_score(y_test, y_pred)
    c8=f1_score(y_test, y_pred)
    return a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6,a7,b7,c7,a8,b8,c8


# In[ ]:


resultsss = []
for i in range(10):
    result = s()
    resultsss.append(result)
print("Results:", resultsss)


# In[ ]:


df = pd.read_csv("C:/Users/gebruiker/Desktop/New folder/marketing_campaign.csv", sep='\t')
df["Income"] = df.groupby("Education")['Income'].transform(lambda x: x.fillna(x.mean()))
df.drop(['Z_CostContact', 'Z_Revenue' , "ID"], axis=1, inplace=True)
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
df['Age'] = df.Year_Birth.apply(lambda x: 2023 - int(x))
df.drop(columns=["Year_Birth"],axis=1 , inplace=True)
df['Education'].replace(['PhD', 'Master','2n Cycle'],'PostGraduate', inplace=True)  
df['Education'].replace(['Graduation'], 'Graduate', inplace=True)
df['Education'].replace(['Basic'],'UnderGraduate', inplace=True)
df["Children"]=df["Kidhome"]+df["Teenhome"]
df.drop(columns=["Kidhome" , "Teenhome"],axis=1 , inplace=True)
categorical_data = df[['Education', 'Marital_Status']]
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat",
                      "MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
df=df.rename(columns={"NumDealsPurchases": "Deals","NumWebPurchases":"Web","NumCatalogPurchases":"Catalog",
                      "NumStorePurchases":"Store","NumWebVisitsMonth":"WebVisits"})
df=df.rename(columns={"AcceptedCmp1": "A1","AcceptedCmp2":"A2","AcceptedCmp3":"A3","AcceptedCmp4":"A4","AcceptedCmp5":"A5"})
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = df['Dt_Customer'].max()
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Time_Enrolled_Days'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df=df.drop(columns=['Dt_Customer','Date_Collected'],axis=1)
df['ActiveDays'] = df['Time_Enrolled_Days'] - df['Recency']
df["Family_Size"] = df["Marital_Status"].replace({"Single": 1, "Couple":2}) + df["Children"]
df.drop(df[(df['Income'] > (df['Income'].mean() * 2)) | (df['Age'] > (df['Age'].mean() * 2))].index, inplace=True)
df['AM.Total'] = df['Wines']+df['Fruits']+df['Meat']+df['Fish']+df['Sweets']+df['Gold']
df['AllPurchase'] = df['Web']+df['Catalog']+df['Store']
df['Average.P'] = round((df['AM.Total'] / df['AllPurchase']), 1)
df['ShareDealsPurchase'] = round((df['Deals'] / df['AllPurchase']) * 100, 1)
df['TotalAcceptedCmp'] = df['A1']+df['A2']+df['A3']+df['A4']+df['A5']
df=df.drop(columns=['A1','A2','A3','A4','A5'],axis=1)
df = df[df["AllPurchase"] != 0]
df.drop(df[df['Average.P'] > (df['Average.P'].mean() * 4)].index, inplace=True)
df.drop(columns=["Average.P"],axis=1 , inplace=True)
print(df.columns)
X= df.drop('Response', axis=1).values
y =df['Response'].values
ct = ColumnTransformer([
    ('catagoric1', OneHotEncoder(), [0]),
    ('catagoric2', OneHotEncoder(), [1]),
    ('numeric', StandardScaler(), list(range(2, len(X.T))))])
scaled_ds = ct.fit_transform(X)
print('Preprocessed Data:')
print(scaled_ds[0])
print(scaled_ds.shape)


# In[ ]:


def f1():
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    param_dist = {'C': [0.01, 0.1, 1 , 10 , 100],
            'kernel': ['poly', 'rbf' , 'linear' , 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'coef0': [-1, 0, 1]}

    model = SVC()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist)
    random_search.fit(X_train_labeled, y_train_labeled)
    modelSVC = SVC(**random_search.best_params_, probability=True)
    modelSVC.fit(X_train_labeled, y_train_labeled)
    y_pred = modelSVC.predict(X_test)
    auc_scores = roc_auc_score(y_test, y_pred)
    F1s = precision_score(y_test, y_pred)
    F1ss = f1_score(y_test, y_pred)
          
    x_values = np.arange(0.03, 0.50, 0.03)
    x_values = np.append(x_values, 0.49)
    scores = np.empty((x_values.shape[0],))
    scoresf =np.empty((x_values.shape[0],))
    scoresac =np.empty((x_values.shape[0],))
    for i, j in enumerate(x_values):
        df_train_main.loc[df_train_main.sample(frac=j).index, 'Random_Mask'] = False
        df_train_main['New_Response']= df_train_main.apply(lambda x:x['Response']if x['Random_Mask'] == False else -1,axis=1)
        X_train = df_train_main.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
        df_train_labeled=df_train[df_train['New_Response']!=-1]
        param_dist = {'C': [0.01, 0.1, 1 , 10 , 100],
            'kernel': ['poly', 'rbf' , 'linear' , 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'coef0': [-1, 0, 1]}

        model = SVC()
        random_search = RandomizedSearchCV(estimator=model,
        param_distributions=param_dist)
        random_search.fit(X_train_labeled, y_train_labeled)        
        modelSVC = SVC(**random_search.best_params_ , probability=True)       
        y_train = df_train_main['New_Response']
        
        x_t = np.arange(0.4, 1.05, 0.05)
        x_t = np.append(x_values, 0.99999)
        scorest = np.empty((x_values.shape[0],))
        for k, threshold in enumerate(x_t):
            self_training_clf = SelfTrainingClassifier(modelSVC, threshold=threshold)
            self_training_clf.fit(X_train, y_train)
            y_predt = self_training_clf.predict(X_test)
            scorest[i] = roc_auc_score(y_test, y_pred)
        max_scoret = np.max(scores)
        max_threshold = x_values[np.argmax(scores)]  
        self_training_clf = SelfTrainingClassifier(modelSVC, threshold=max_threshold)
        self_training_clf.fit(X_train, y_train)
        y_pred = self_training_clf.predict(X_test)
        scores[i] = roc_auc_score(y_test, y_pred)
        scoresf[i] = precision_score(y_test, y_pred)
        scoresac[i] = f1_score(y_test, y_pred)
    return (x_values , scores , scoresf , scoresac , auc_scores , F1s ,  F1ss , X_test, y_test)


# In[ ]:


results1 = []
for i in range(10):
    result = f1()
    results1.append(result)
print("Results:", results1)


# In[ ]:


mn_scoresA = np.mean([results1[i][1] for i in range(10)], axis=0)
mn_scoresFB = np.mean([results1[i][2] for i in range(10)], axis=0)
mn_scoresF1 = np.mean([results1[i][3] for i in range(10)], axis=0)
std_score = np.std([results1[i][3] for i in range(10)], axis=0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results1[0][0], results1[0][3], label='Scores1')
ax.plot(results1[0][0], results1[1][3], label='Scores2')
ax.plot(results1[0][0], results1[2][3], label='Scores3')
ax.plot(results1[0][0], results1[3][3], label='Scores4')
ax.plot(results1[0][0], results1[4][3], label='Scores5')
ax.plot(results1[0][0], results1[5][3], label='Scores6')
ax.plot(results1[0][0], results1[6][3], label='Scores7')
ax.plot(results1[0][0], results1[7][3], label='Scores8')
ax.plot(results1[0][0], results1[8][3], label='Scores9')
ax.plot(results1[0][0], results1[9][3], label='Scores10')
ax.fill_between(results1[0][0], mn_scoresF1-std_score, mn_scoresF1+std_score, alpha=0.2)
ax.legend()
ax.set_title('Shuffle Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('Accuracy score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCf = np.mean([i[3] for i in results1], axis=0)
for i in range(len(results1[0][0])):
    mean_SVCsupervised = np.mean([i[6] for i in results1], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[3] for i in results1], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results1[0][0], mean_SVCf, label='self-training F1' , color='red')
ax.fill_between(results1[0][0], mean_SVCf-std_score/2, mean_SVCf+std_score/2, alpha=0.1 , color='red')
ax.plot(results1[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('SVM')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('F1 score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB)]
z=[]
mean_SVCp = np.mean([i[2] for i in results1], axis=0)
for i in range(len(results1[0][0])):
    mean_SVCsupervised = np.mean([i[5] for i in results1], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[2] for i in results1], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results1[0][0], mean_SVCp, label='self-training F1' , color='blue')
ax.fill_between(results1[0][0], mean_SVCp-std_score/2, mean_SVCp+std_score/2, alpha=0.1 , color='blue')
ax.plot(results1[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('SVM')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('precision score')
plt.show()


# In[ ]:



mean_SVCf = np.mean([i[3] for i in results1], axis=0)
mean_SVCp = np.mean([i[2] for i in results1], axis=0)
std_SVCf = np.std([i[3] for i in results1], axis=0)
std_SVCp = np.std([i[2] for i in results1], axis=0)

w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCr1=(mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))
for i in range(len(results1[0][0])):
    mean_SVCsupervised = np.mean([(i[6]*i[5]/((2*i[5])-i[6])) for i in results1], axis=0)
    z.append(mean_SVCsupervised)
std_score=(std_SVCf*std_SVCp/((2*std_SVCp)-std_SVCf))

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results1[0][0], mean_SVCr1, label='self-training recall' , color='red')
ax.fill_between(results1[0][0], mean_SVCr1-std_score/2, mean_SVCr1+std_score/2, alpha=0.1 , color='red')
ax.plot(results1[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('SVM')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('recall score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresA)]

z=[]
mean_SVC = np.mean([i[1] for i in results1], axis=0)
for i in range(len(results1[0][0])):
    mean_SVCsupervised = np.mean([i[4] for i in results1], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[1] for i in results1], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results1[0][0], mean_SVC, label='self-training ROC-AUC' , color='brown')
ax.fill_between(results1[0][0], mean_SVC-std_score/2, mean_SVC+std_score/2, alpha=0.1 , color='brown')
ax.plot(results1[0][0], z, label='supervised ROC-AUC ', color='green', linestyle='--')

ax.legend()
ax.set_title('SVM')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('ROC-AUC score')
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X_train, y_train)
df_train = pd.DataFrame(X_bal)
df_train['Response'] = y_bal
df_train['Random_Mask'] = True
df_train['New_Response']=df_train['Response']
df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
df_train_labeled=df_train[df_train['New_Response']!=-1]        
X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
y_train_labeled = df_train_labeled['New_Response']
param_dist = {'C': [0.01, 0.1, 1 , 10 , 100],
        'kernel': ['poly', 'rbf' , 'linear' , 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'coef0': [-1, 0, 1]}

model = SVC()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modelSVC = SVC(**random_search.best_params_, probability=True)
modelSVC.fit(X_train_labeled, y_train_labeled)
plot_confusion_matrix(modelSVC, X_test, y_test, cmap='Blues', normalize='true',
                 display_labels=['Response', 'No Response']);


# In[ ]:


a=[]
b=[]
c=[]
mean_listauc = [] 
mean_listf1 = [] 
mean_listacc = [] 
for i in range(10):
    mean_val1 = (results1[i][1][7])
    mean_val2 = (results1[i][2][7])
    mean_val3 = (results1[i][3][7])
    aa=resultsss[i][0]
    bb=resultsss[i][1]
    cc=resultsss[i][2]
    a.append(aa)
    b.append(bb)
    c.append(cc)   
    mean_listauc.append(mean_val1)
    mean_listf1.append(mean_val3)
    mean_listacc.append(mean_val2)
    
mean_supervisedauc = np.mean(a)
mean_supervisedpre = np.mean(b)
mean_supervisedf1 = np.mean(c)
    
mean_self_trainauc=np.mean(mean_listauc)
mean_self_trainfpre=np.mean(mean_listacc)
mean_self_trainf1=np.mean(mean_listf1)


print("Self-training AUC score with SVM classifier is:", "{:.3f}%".format(mean_self_trainauc * 100))
print("Supervised logistic classifier model AUC score is:", "{:.3f}%".format(mean_supervisedauc * 100))
if mean_self_trainauc > mean_supervisedauc:
    print("Self-training AUC score with SVM classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training AUC score with SVM classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training precision with SVM classifier is:", "{:.3f}%".format(mean_self_trainfpre * 100))
print("Supervised SVM model precision is:", "{:.3f}%".format(mean_supervisedpre * 100))
if mean_self_trainfpre > mean_supervisedpre:
    print("Self-training precision with SVM classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training precision with SVM classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training F1 with logistic SVM is:", "{:.3f}%".format(mean_self_trainf1 * 100))
print("Supervised SVM model accuracy is:", "{:.3f}%".format(mean_supervisedf1 * 100))
if mean_self_trainf1 > mean_supervisedf1:
    print("Self-training F1 with SVM classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training F1 with SVM classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training recall with SVM is:", "{:.3f}%".format(((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1)))  * 100))
print("Supervised logistic model recall is:", "{:.3f}%".format(((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))* 100))
if (((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1))))  > (((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))):
    print("Self-training recall with SVM  is greater than supervised model for 24% of labeled data")
else:
    print("Self-training recall with SVM  is not greater than supervised model for 24% of labeled data")


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB, mn_scoresF1,mn_scoresA)]
mean_SVC = np.mean([i[1] for i in results1], axis=0)
    
max_index = np.argmax(w[0][1]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][1][max_index]
print("maximum  precision of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum precision:", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][2]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][2][max_index]
print("maximum F1 of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum F1 :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][3]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][3][max_index]
print("maximum AUC  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum AUC :", "{:.1f}%".format(max_value_first_row*100))


max_index = np.argmax((mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = (mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))[max_index]
print("maximum recall  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum recall :", "{:.1f}%".format(max_value_first_row*100))


# In[ ]:


df = pd.read_csv("C:/Users/gebruiker/Desktop/New folder/marketing_campaign.csv", sep='\t')
df["Income"] = df.groupby("Education")['Income'].transform(lambda x: x.fillna(x.mean()))
df.drop(['Z_CostContact', 'Z_Revenue' , "ID"], axis=1, inplace=True)
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
df['Age'] = df.Year_Birth.apply(lambda x: 2023 - int(x))
df.drop(columns=["Year_Birth"],axis=1 , inplace=True)
df['Education'].replace(['PhD', 'Master','2n Cycle'],'PostGraduate', inplace=True)  
df['Education'].replace(['Graduation'], 'Graduate', inplace=True)
df['Education'].replace(['Basic'],'UnderGraduate', inplace=True)
df["Children"]=df["Kidhome"]+df["Teenhome"]
df.drop(columns=["Kidhome" , "Teenhome"],axis=1 , inplace=True)
categorical_data = df[['Education', 'Marital_Status']]
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat",
                      "MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
df=df.rename(columns={"NumDealsPurchases": "Deals","NumWebPurchases":"Web","NumCatalogPurchases":
                      "Catalog","NumStorePurchases":"Store","NumWebVisitsMonth":"WebVisits"})
df=df.rename(columns={"AcceptedCmp1": "A1","AcceptedCmp2":"A2","AcceptedCmp3":"A3","AcceptedCmp4":"A4","AcceptedCmp5":"A5"})
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = df['Dt_Customer'].max()
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Time_Enrolled_Days'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df=df.drop(columns=['Dt_Customer','Date_Collected'],axis=1)
df['ActiveDays'] = df['Time_Enrolled_Days'] - df['Recency']
df["Family_Size"] = df["Marital_Status"].replace({"Single": 1, "Couple":2}) + df["Children"]
df.drop(df[(df['Income'] > (df['Income'].mean() * 2)) | (df['Age'] > (df['Age'].mean() * 2))].index, inplace=True)
df['AM.Total'] = df['Wines']+df['Fruits']+df['Meat']+df['Fish']+df['Sweets']+df['Gold']
df['AllPurchase'] = df['Web']+df['Catalog']+df['Store']
df['Average.P'] = round((df['AM.Total'] / df['AllPurchase']), 1)
df['ShareDealsPurchase'] = round((df['Deals'] / df['AllPurchase']) * 100, 1)
df['TotalAcceptedCmp'] = df['A1']+df['A2']+df['A3']+df['A4']+df['A5']
df=df.drop(columns=['A1','A2','A3','A4','A5'],axis=1)
df = df[df["AllPurchase"] != 0]
df.drop(df[df['Average.P'] > (df['Average.P'].mean() * 4)].index, inplace=True)
df.drop(columns=["Average.P"],axis=1 , inplace=True)
X= df.drop('Response', axis=1).values
y =df['Response'].values
ct = ColumnTransformer([
    ('catagoric1', OneHotEncoder(), [0]),
    ('catagoric2', OneHotEncoder(), [1]),
    ('numeric', StandardScaler(), list(range(2, len(X.T))))])
scaled_ds = ct.fit_transform(X)
print('Preprocessed Data:')
print(scaled_ds[0])
print(scaled_ds.shape)


# In[ ]:


def f2():
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    param_dist =  {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'max_iter': [100, 1000, 5000]}

    model = LogisticRegression()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist)
    random_search.fit(X_train_labeled, y_train_labeled)
    modellog = LogisticRegression(**random_search.best_params_)
    modellog.fit(X_train_labeled, y_train_labeled)
    y_pred = modellog.predict(X_test)
    auc_scores = roc_auc_score(y_test, y_pred)
    F1s = precision_score(y_test, y_pred)
    F1ss = f1_score(y_test, y_pred)
          
    x_values = np.arange(0.03, 0.50, 0.03)
    x_values = np.append(x_values, 0.49)
    scores = np.empty((x_values.shape[0],))
    scoresf =np.empty((x_values.shape[0],))
    scoresac =np.empty((x_values.shape[0],))
    for i, j in enumerate(x_values):
        df_train_main.loc[df_train_main.sample(frac=j).index, 'Random_Mask'] = False
        df_train_main['New_Response']= df_train_main.apply(lambda x:x['Response']if x['Random_Mask'] == False else -1,axis=1)
        X_train = df_train_main.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
        df_train_labeled=df_train[df_train['New_Response']!=-1]
        param_dist =  {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'max_iter': [100, 1000, 5000]}
        model = LogisticRegression()
        random_search = RandomizedSearchCV(estimator=model,
        param_distributions=param_dist)
        random_search.fit(X_train_labeled, y_train_labeled)        
        modellog = LogisticRegression(**random_search.best_params_ )       
        y_train = df_train_main['New_Response']
        
        x_t = np.arange(0.4, 1.05, 0.05)
        x_t = np.append(x_values, 0.99999)
        scorest = np.empty((x_values.shape[0],))
        for k, threshold in enumerate(x_t):
            self_training_clf = SelfTrainingClassifier(modellog, threshold=threshold)
            self_training_clf.fit(X_train, y_train)
            y_predt = self_training_clf.predict(X_test)
            scorest[i] = roc_auc_score(y_test, y_pred)
        max_scoret = np.max(scores)
        max_threshold = x_values[np.argmax(scores)]  
        self_training_clf = SelfTrainingClassifier(modellog, threshold=max_threshold)
        self_training_clf.fit(X_train, y_train)
        y_pred = self_training_clf.predict(X_test)
        scores[i] = roc_auc_score(y_test, y_pred)
        scoresf[i] = precision_score(y_test, y_pred)
        scoresac[i] = f1_score(y_test, y_pred)
    return (x_values , scores , scoresf , scoresac , auc_scores , F1s ,  F1ss , X_test, y_test)


# In[ ]:


results2 = []
for i in range(10):
    result = f2()
    results2.append(result)
print("Results:", results2)


# In[ ]:


mn_scoresA = np.mean([results2[i][1] for i in range(10)], axis=0)
mn_scoresFB = np.mean([results2[i][2] for i in range(10)], axis=0)
mn_scoresF1 = np.mean([results2[i][3] for i in range(10)], axis=0)
std_score = np.std([results2[i][3] for i in range(10)], axis=0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results2[0][0], results2[0][3], label='Scores1')
ax.plot(results2[0][0], results2[1][3], label='Scores2')
ax.plot(results2[0][0], results2[2][3], label='Scores3')
ax.plot(results2[0][0], results2[3][3], label='Scores4')
ax.plot(results2[0][0], results2[4][3], label='Scores5')
ax.plot(results2[0][0], results2[5][3], label='Scores6')
ax.plot(results2[0][0], results2[6][3], label='Scores7')
ax.plot(results2[0][0], results2[7][3], label='Scores8')
ax.plot(results2[0][0], results2[8][3], label='Scores9')
ax.plot(results2[0][0], results2[9][3], label='Scores10')
ax.fill_between(results1[0][0], mn_scoresF1-std_score, mn_scoresF1+std_score, alpha=0.2)
ax.legend()
ax.set_title('Shuffle Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('Accuracy score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCf = np.mean([i[3] for i in results2], axis=0)
for i in range(len(results2[0][0])):
    mean_SVCsupervised = np.mean([i[6] for i in results2], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[3] for i in results2], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results2[0][0], mean_SVCf, label='self-training F1' , color='red')
ax.fill_between(results2[0][0], mean_SVCf-std_score/2, mean_SVCf+std_score/2, alpha=0.1 , color='red')
ax.plot(results2[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Logistic Regresion')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('F1 score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCp = np.mean([i[2] for i in results2], axis=0)
for i in range(len(results2[0][0])):
    mean_SVCsupervised = np.mean([i[5] for i in results2], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[2] for i in results2], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results2[0][0], mean_SVCp, label='self-training F1' , color='blue')
ax.fill_between(results2[0][0], mean_SVCp-std_score/2, mean_SVCp+std_score/2, alpha=0.1 , color='blue')
ax.plot(results2[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Logistic Regresion')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('precision score')
plt.show()


# In[ ]:


mean_SVCf = np.mean([i[3] for i in results2], axis=0)
mean_SVCp = np.mean([i[2] for i in results2], axis=0)
std_SVCf = np.std([i[3] for i in results2], axis=0)
std_SVCp = np.std([i[2] for i in results2], axis=0)

w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCr2=(mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))
for i in range(len(results2[0][0])):
    mean_SVCsupervised = np.mean([(i[6]*i[5]/((2*i[5])-i[6])) for i in results2], axis=0)
    z.append(mean_SVCsupervised)
std_score=(std_SVCf*std_SVCp/((2*std_SVCp)-std_SVCf))

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results2[0][0], mean_SVCr2, label='self-training recall' , color='red')
ax.fill_between(results2[0][0], mean_SVCr2-std_score/2, mean_SVCr2+std_score/2, alpha=0.1 , color='red')
ax.plot(results2[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('SVC')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('recall score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVC = np.mean([i[1] for i in results2], axis=0)
for i in range(len(results2[0][0])):
    mean_SVCsupervised = np.mean([i[4] for i in results2], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[1] for i in results2], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results2[0][0], mean_SVC, label='self-training ROC-AUC' , color='brown')
ax.fill_between(results2[0][0], mean_SVC-std_score/2, mean_SVC+std_score/2, alpha=0.1 , color='brown')
ax.plot(results2[0][0], z, label='supervised ROC-AUC ', color='green', linestyle='--')

ax.legend()
ax.set_title('Logistic Regresion')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('ROC-AUC score')
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X_train, y_train)
df_train = pd.DataFrame(X_bal)
df_train['Response'] = y_bal
df_train['Random_Mask'] = True
df_train['New_Response']=df_train['Response']
df_train_main=df_train.copy()
df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
df_train_labeled=df_train[df_train['New_Response']!=-1]        
X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
y_train_labeled = df_train_labeled['New_Response']
param_dist =  {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'max_iter': [100, 1000, 5000]}

model = LogisticRegression()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modellog = LogisticRegression(**random_search.best_params_)
modellog.fit(X_train_labeled, y_train_labeled)
plot_confusion_matrix(modellog, X_test, y_test, cmap='Blues', normalize='true',
                 display_labels=['Response', 'No Response']);


# In[ ]:


a=[]
b=[]
c=[]
mean_listauc = [] 
mean_listf1 = [] 
mean_listacc = [] 
for i in range(10):
    mean_val1 = (results2[i][1][7])
    mean_val2 = (results2[i][2][7])
    mean_val3 = (results2[i][3][7])
    aa=resultsss[i][3]
    bb=resultsss[i][4]
    cc=resultsss[i][5]
    a.append(aa)
    b.append(bb)
    c.append(cc)   
    mean_listauc.append(mean_val1)
    mean_listf1.append(mean_val3)
    mean_listacc.append(mean_val2)
    
mean_supervisedauc = np.mean(a)
mean_supervisedpre = np.mean(b)
mean_supervisedf1 = np.mean(c)
    
mean_self_trainauc=np.mean(mean_listauc)
mean_self_trainfpre=np.mean(mean_listacc)
mean_self_trainf1=np.mean(mean_listf1)


print("Self-training AUC score with logistic classifier classifier is:", "{:.3f}%".format(mean_self_trainauc * 100))
print("Supervised logistic classifier model AUC score is:", "{:.3f}%".format(mean_supervisedauc * 100))
if mean_self_trainauc > mean_supervisedauc:
    print("Self-training AUC score with logistic classifier classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training AUC score with logistic classifier classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training precision with logistic classifier classifier is:", "{:.3f}%".format(mean_self_trainfpre * 100))
print("Supervised logistic classifier model precision is:", "{:.3f}%".format(mean_supervisedpre * 100))
if mean_self_trainfpre > mean_supervisedpre:
    print("Self-training precision with logistic classifier classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training precision with logistic classifier classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training F1 with logistic classifier classifier is:", "{:.3f}%".format(mean_self_trainf1 * 100))
print("Supervised logistic classifier model accuracy is:", "{:.3f}%".format(mean_supervisedf1 * 100))
if mean_self_trainf1 > mean_supervisedf1:
    print("Self-training F1 with logistic classifier classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training F1 with logistic classifier classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training recall with logistic classifier is:", "{:.3f}%".format(((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1)))  * 100))
print("Supervised logistic model recall is:", "{:.3f}%".format(((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))* 100))
if (((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1))))  > (((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))):
    print("Self-training recall with logistic  is greater than supervised model for 24% of labeled data")
else:
    print("Self-training recall with logistic  is not greater than supervised model for 24% of labeled data")


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB, mn_scoresF1,mn_scoresA)]
mean_SVC = np.mean([i[1] for i in results2], axis=0)
    
max_index = np.argmax(w[0][1]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][1][max_index]
print("maximum  precision of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum precision:", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][2]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][2][max_index]
print("maximum F1 of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum F1 :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][3]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][3][max_index]
print("maximum AUC  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum AUC :", "{:.1f}%".format(max_value_first_row*100))


max_index = np.argmax((mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = (mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))[max_index]
print("maximum recall  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum recall :", "{:.1f}%".format(max_value_first_row*100))


# In[ ]:


df = pd.read_csv("C:/Users/gebruiker/Desktop/New folder/marketing_campaign.csv", sep='\t')
df["Income"] = df.groupby("Education")['Income'].transform(lambda x: x.fillna(x.mean()))
df.drop(['Z_CostContact', 'Z_Revenue' , "ID"], axis=1, inplace=True)
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
df['Age'] = df.Year_Birth.apply(lambda x: 2023 - int(x))
df.drop(columns=["Year_Birth"],axis=1 , inplace=True)
df['Education'].replace(['PhD', 'Master','2n Cycle'],'PostGraduate', inplace=True)  
df['Education'].replace(['Graduation'], 'Graduate', inplace=True)
df['Education'].replace(['Basic'],'UnderGraduate', inplace=True)
df["Children"]=df["Kidhome"]+df["Teenhome"]
df.drop(columns=["Kidhome" , "Teenhome"],axis=1 , inplace=True)
categorical_data = df[['Education', 'Marital_Status']]
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
df=df.rename(columns={"NumDealsPurchases": "Deals","NumWebPurchases":"Web","NumCatalogPurchases":"Catalog","NumStorePurchases":"Store","NumWebVisitsMonth":"WebVisits"})
df=df.rename(columns={"AcceptedCmp1": "A1","AcceptedCmp2":"A2","AcceptedCmp3":"A3","AcceptedCmp4":"A4","AcceptedCmp5":"A5"})
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = df['Dt_Customer'].max()
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Time_Enrolled_Days'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df=df.drop(columns=['Dt_Customer','Date_Collected'],axis=1)
df['ActiveDays'] = df['Time_Enrolled_Days'] - df['Recency']
df["Family_Size"] = df["Marital_Status"].replace({"Single": 1, "Couple":2}) + df["Children"]
df.drop(df[(df['Income'] > (df['Income'].mean() * 2)) | (df['Age'] > (df['Age'].mean() * 2))].index, inplace=True)
df['AM.Total'] = df['Wines']+df['Fruits']+df['Meat']+df['Fish']+df['Sweets']+df['Gold']
df['AllPurchase'] = df['Web']+df['Catalog']+df['Store']
df['Average.P'] = round((df['AM.Total'] / df['AllPurchase']), 1)
df['ShareDealsPurchase'] = round((df['Deals'] / df['AllPurchase']) * 100, 1)
df['TotalAcceptedCmp'] = df['A1']+df['A2']+df['A3']+df['A4']+df['A5']
df=df.drop(columns=['A1','A2','A3','A4','A5'],axis=1)
df = df[df["AllPurchase"] != 0]
df.drop(df[df['Average.P'] > (df['Average.P'].mean() * 4)].index, inplace=True)
df.drop(columns=["Average.P"],axis=1 , inplace=True)
X= df.drop('Response', axis=1).values
y =df['Response'].values
ct = ColumnTransformer([
    ('catagoric1', OneHotEncoder(), [0]),
    ('catagoric2', OneHotEncoder(), [1]),
    ('numeric', StandardScaler(), list(range(2, len(X.T))))])
scaled_ds = ct.fit_transform(X)
print('Preprocessed Data:')
print(scaled_ds[0])
print(scaled_ds.shape)


# In[ ]:


def f3():
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    param_dist =  {
    'n_estimators': [50, 100, 200, 300],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']}

    model = RandomForestClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist)
    random_search.fit(X_train_labeled, y_train_labeled)
    modelrfs = RandomForestClassifier(**random_search.best_params_)
    modelrfs.fit(X_train_labeled, y_train_labeled)
    y_pred = modelrfs.predict(X_test)
    auc_scores = roc_auc_score(y_test, y_pred)
    F1s = precision_score(y_test, y_pred)
    F1ss = f1_score(y_test, y_pred)
          
    x_values = np.arange(0.03, 0.50, 0.03)
    x_values = np.append(x_values, 0.49)
    scores = np.empty((x_values.shape[0],))
    scoresf =np.empty((x_values.shape[0],))
    scoresac =np.empty((x_values.shape[0],))
    for i, j in enumerate(x_values):
        df_train_main.loc[df_train_main.sample(frac=j).index, 'Random_Mask'] = False
        df_train_main['New_Response']= df_train_main.apply(lambda x:x['Response']if x['Random_Mask'] == False else -1,axis=1)
        X_train = df_train_main.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
        df_train_labeled=df_train[df_train['New_Response']!=-1]
        param_dist =  {
        'n_estimators': [50, 100, 200, 300],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']}

        model = RandomForestClassifier()
        random_search = RandomizedSearchCV(estimator=model,
        param_distributions=param_dist)
        random_search.fit(X_train_labeled, y_train_labeled)        
        modelrfs = RandomForestClassifier(**random_search.best_params_ )       
        y_train = df_train_main['New_Response']
        
        x_t = np.arange(0.4, 1.05, 0.05)
        x_t = np.append(x_values, 0.99999)
        scorest = np.empty((x_values.shape[0],))
        for k, threshold in enumerate(x_t):
            self_training_clf = SelfTrainingClassifier(modelrfs, threshold=threshold)
            self_training_clf.fit(X_train, y_train)
            y_predt = self_training_clf.predict(X_test)
            scorest[i] = roc_auc_score(y_test, y_pred)
        max_scoret = np.max(scores)
        max_threshold = x_values[np.argmax(scores)]  
        self_training_clf = SelfTrainingClassifier(modelrfs, threshold=max_threshold)
        self_training_clf.fit(X_train, y_train)
        y_pred = self_training_clf.predict(X_test)
        scores[i] = roc_auc_score(y_test, y_pred)
        scoresf[i] = precision_score(y_test, y_pred)
        scoresac[i] = f1_score(y_test, y_pred)
    return (x_values , scores , scoresf , scoresac , auc_scores , F1s ,  F1ss , X_test, y_test)


# In[ ]:


results3 = []
for i in range(10):
    result = f3()
    results3.append(result)
print("Results:", results3)


# In[ ]:


mn_scoresA = np.mean([results3[i][1] for i in range(10)], axis=0)
mn_scoresFB = np.mean([results3[i][2] for i in range(10)], axis=0)
mn_scoresF1 = np.mean([results3[i][3] for i in range(10)], axis=0)
std_score = np.std([results3[i][3] for i in range(10)], axis=0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results3[0][0], results3[0][3], label='Scores1')
ax.plot(results3[0][0], results3[1][3], label='Scores2')
ax.plot(results3[0][0], results3[2][3], label='Scores3')
ax.plot(results3[0][0], results3[3][3], label='Scores4')
ax.plot(results3[0][0], results3[4][3], label='Scores5')
ax.plot(results3[0][0], results3[5][3], label='Scores6')
ax.plot(results3[0][0], results3[6][3], label='Scores7')
ax.plot(results3[0][0], results3[7][3], label='Scores8')
ax.plot(results3[0][0], results3[8][3], label='Scores9')
ax.plot(results3[0][0], results3[9][3], label='Scores10')
ax.fill_between(results3[0][0], mn_scoresF1-std_score, mn_scoresF1+std_score, alpha=0.2)
ax.legend()
ax.set_title('Shuffle Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('Accuracy score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCf = np.mean([i[3] for i in results3], axis=0)
for i in range(len(results3[0][0])):
    mean_SVCsupervised = np.mean([i[6] for i in results3], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[3] for i in results3], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results3[0][0], mean_SVCf, label='self-training F1' , color='red')
ax.fill_between(results3[0][0], mean_SVCf-std_score/2, mean_SVCf+std_score/2, alpha=0.1 , color='red')
ax.plot(results3[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Randomforest Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('F1 score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCp = np.mean([i[2] for i in results3], axis=0)
for i in range(len(results3[0][0])):
    mean_SVCsupervised = np.mean([i[5] for i in results3], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[2] for i in results3], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results3[0][0], mean_SVCp, label='self-training F1' , color='blue')
ax.fill_between(results3[0][0], mean_SVCp-std_score/2, mean_SVCp+std_score/2, alpha=0.1 , color='blue')
ax.plot(results3[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Randomforest Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('precision score')
plt.show()


# In[ ]:


mean_SVCf = np.mean([i[3] for i in results3], axis=0)
mean_SVCp = np.mean([i[2] for i in results3], axis=0)
std_SVCf = np.std([i[3] for i in results3], axis=0)
std_SVCp = np.std([i[2] for i in results3], axis=0)

print(mean_SVCf,mean_SVCp)

w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCr3=(mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))
for i in range(len(results3[0][0])):
    mean_SVCsupervised = np.mean([(i[6]*i[5]/((2*i[5])-i[6])) for i in results3], axis=0)
    z.append(mean_SVCsupervised)
std_score=(std_SVCf*std_SVCp/((2*std_SVCp)-std_SVCf))

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results3[0][0], mean_SVCr3, label='self-training recall' , color='red')
ax.fill_between(results3[0][0], mean_SVCr3-std_score/2, mean_SVCr3+std_score/2, alpha=0.1 , color='red')
ax.plot(results3[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('SVC')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('recall score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVC = np.mean([i[1] for i in results3], axis=0)
for i in range(len(results1[0][0])):
    mean_SVCsupervised = np.mean([i[4] for i in results3], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[1] for i in results3], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results1[0][0], mean_SVC, label='self-training ROC-AUC' , color='brown')
ax.fill_between(results1[0][0], mean_SVC-std_score/2, mean_SVC+std_score/2, alpha=0.1 , color='brown')
ax.plot(results1[0][0], z, label='supervised ROC-AUC ', color='green', linestyle='--')

ax.legend()
ax.set_title('Randomforest')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('ROC-AUC score')
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X_train, y_train)
df_train = pd.DataFrame(X_bal)
df_train['Response'] = y_bal
df_train['Random_Mask'] = True
df_train['New_Response']=df_train['Response']
df_train_main=df_train.copy()
df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
df_train_labeled=df_train[df_train['New_Response']!=-1]        
X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
y_train_labeled = df_train_labeled['New_Response']
param_dist =  {
'n_estimators': [50, 100, 200, 300],
'min_samples_split': [5, 10],
'min_samples_leaf': [1, 2, 4],
'criterion': ['gini', 'entropy']}

model = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modelrfs = RandomForestClassifier(**random_search.best_params_)
modelrfs.fit(X_train_labeled, y_train_labeled)
plot_confusion_matrix(modelrfs, X_test, y_test, cmap='Blues', normalize='true',
                 display_labels=['Response', 'No Response']);


# In[ ]:


a=[]
b=[]
c=[]
mean_listauc = [] 
mean_listf1 = [] 
mean_listacc = [] 
for i in range(10):
    mean_val1 = (results3[i][1][7])
    mean_val2 = (results3[i][2][7])
    mean_val3 = (results3[i][3][7])
    aa=resultsss[i][6]
    bb=resultsss[i][7]
    cc=resultsss[i][8]
    a.append(aa)
    b.append(bb)
    c.append(cc)   
    mean_listauc.append(mean_val1)
    mean_listf1.append(mean_val3)
    mean_listacc.append(mean_val2)
    
mean_supervisedauc = np.mean(a)
mean_supervisedpre = np.mean(b)
mean_supervisedf1 = np.mean(c)
    
mean_self_trainauc=np.mean(mean_listauc)
mean_self_trainfpre=np.mean(mean_listacc)
mean_self_trainf1=np.mean(mean_listf1)


print("Self-training AUC score with RANDOMFOREST classifier is:", "{:.3f}%".format(mean_self_trainauc * 100))
print("Supervised RANDOMFOREST model AUC score is:", "{:.3f}%".format(mean_supervisedauc * 100))
if mean_self_trainauc > mean_supervisedauc:
    print("Self-training AUC score with RANDOMFOREST classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training AUC score with RANDOMFOREST classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training precision with RANDOMFOREST classifier is:", "{:.3f}%".format(mean_self_trainfpre * 100))
print("Supervised RANDOMFOREST model precision is:", "{:.3f}%".format(mean_supervisedpre * 100))
if mean_self_trainfpre > mean_supervisedpre:
    print("Self-training precision with RANDOMFOREST classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training precision with RANDOMFOREST classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training F1 with logistic RANDOMFOREST is:", "{:.3f}%".format(mean_self_trainf1 * 100))
print("Supervised RANDOMFOREST model accuracy is:", "{:.3f}%".format(mean_supervisedf1 * 100))
if mean_self_trainf1 > mean_supervisedf1:
    print("Self-training F1 with RANDOMFOREST classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training F1 with RANDOMFOREST classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training recall with RANDOMFOREST is:", "{:.3f}%".format(((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1)))  * 100))
print("Supervised RANDOMFOREST model recall is:", "{:.3f}%".format(((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))* 100))
if (((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1))))  > (((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))):
    print("Self-training recall with RANDOMFOREST  is greater than supervised model for 24% of labeled data")
else:
    print("Self-training recall with RANDOMFOREST  is not greater than supervised model for 24% of labeled data")


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB, mn_scoresF1,mn_scoresA)]
mean_SVC = np.mean([i[1] for i in results3], axis=0)
    
max_index = np.argmax(w[0][1]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][1][max_index]
print("maximum  precision of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum precision:", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][2]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][2][max_index]
print("maximum F1 of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum F1 :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][3]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][3][max_index]
print("maximum AUC  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum AUC :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax((mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = (mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))[max_index]
print("maximum recall  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum recall :", "{:.1f}%".format(max_value_first_row*100))


# In[ ]:


df = pd.read_csv("C:/Users/gebruiker/Desktop/New folder/marketing_campaign.csv", sep='\t')
df["Income"] = df.groupby("Education")['Income'].transform(lambda x: x.fillna(x.mean()))
df.drop(['Z_CostContact', 'Z_Revenue' , "ID"], axis=1, inplace=True)
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
df['Age'] = df.Year_Birth.apply(lambda x: 2023 - int(x))
df.drop(columns=["Year_Birth"],axis=1 , inplace=True)
df['Education'].replace(['PhD', 'Master','2n Cycle'],'PostGraduate', inplace=True)  
df['Education'].replace(['Graduation'], 'Graduate', inplace=True)
df['Education'].replace(['Basic'],'UnderGraduate', inplace=True)
df["Children"]=df["Kidhome"]+df["Teenhome"]
df.drop(columns=["Kidhome" , "Teenhome"],axis=1 , inplace=True)
categorical_data = df[['Education', 'Marital_Status']]
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
df=df.rename(columns={"NumDealsPurchases": "Deals","NumWebPurchases":"Web","NumCatalogPurchases":"Catalog","NumStorePurchases":"Store","NumWebVisitsMonth":"WebVisits"})
df=df.rename(columns={"AcceptedCmp1": "A1","AcceptedCmp2":"A2","AcceptedCmp3":"A3","AcceptedCmp4":"A4","AcceptedCmp5":"A5"})
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = df['Dt_Customer'].max()
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Time_Enrolled_Days'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df=df.drop(columns=['Dt_Customer','Date_Collected'],axis=1)
df['ActiveDays'] = df['Time_Enrolled_Days'] - df['Recency']
df["Family_Size"] = df["Marital_Status"].replace({"Single": 1, "Couple":2}) + df["Children"]
df.drop(df[(df['Income'] > (df['Income'].mean() * 2)) | (df['Age'] > (df['Age'].mean() * 2))].index, inplace=True)
df['AM.Total'] = df['Wines']+df['Fruits']+df['Meat']+df['Fish']+df['Sweets']+df['Gold']
df['AllPurchase'] = df['Web']+df['Catalog']+df['Store']
df['Average.P'] = round((df['AM.Total'] / df['AllPurchase']), 1)
df['ShareDealsPurchase'] = round((df['Deals'] / df['AllPurchase']) * 100, 1)
df['TotalAcceptedCmp'] = df['A1']+df['A2']+df['A3']+df['A4']+df['A5']
df=df.drop(columns=['A1','A2','A3','A4','A5'],axis=1)
df = df[df["AllPurchase"] != 0]
df.drop(df[df['Average.P'] > (df['Average.P'].mean() * 4)].index, inplace=True)
df.drop(columns=["Average.P"],axis=1 , inplace=True)
X= df.drop('Response', axis=1).values
y =df['Response'].values
ct = ColumnTransformer([
    ('catagoric1', OneHotEncoder(), [0]),
    ('catagoric2', OneHotEncoder(), [1]),
    ('numeric', StandardScaler(), list(range(2, len(X.T))))])
scaled_ds = ct.fit_transform(X)
print('Preprocessed Data:')
print(scaled_ds[0])
print(scaled_ds.shape)


# In[ ]:


def f4():
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    param_dist =  {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-10, 1e-11, 1e-12, 1e-4]}

    model = GaussianNB()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist)
    random_search.fit(X_train_labeled, y_train_labeled)
    modelnb = GaussianNB(**random_search.best_params_)
    modelnb.fit(X_train_labeled, y_train_labeled)
    y_pred = modelnb.predict(X_test)
    auc_scores = roc_auc_score(y_test, y_pred)
    F1s = precision_score(y_test, y_pred)
    F1ss = f1_score(y_test, y_pred)
          
    x_values = np.arange(0.03, 0.50, 0.03)
    x_values = np.append(x_values, 0.49)
    scores = np.empty((x_values.shape[0],))
    scoresf =np.empty((x_values.shape[0],))
    scoresac =np.empty((x_values.shape[0],))
    for i, j in enumerate(x_values):
        df_train_main.loc[df_train_main.sample(frac=j).index, 'Random_Mask'] = False
        df_train_main['New_Response']= df_train_main.apply(lambda x:x['Response']if x['Random_Mask'] == False else -1,axis=1)
        X_train = df_train_main.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
        df_train_labeled=df_train[df_train['New_Response']!=-1]
        param_dist =   {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-10, 1e-11, 1e-12, 1e-4]}
        model = GaussianNB()
        random_search = RandomizedSearchCV(estimator=model,
        param_distributions=param_dist)
        random_search.fit(X_train_labeled, y_train_labeled)        
        modelnb = GaussianNB(**random_search.best_params_ )       
        y_train = df_train_main['New_Response']
        
        x_t = np.arange(0.4, 1.05, 0.05)
        x_t = np.append(x_values, 0.99999)
        scorest = np.empty((x_values.shape[0],))
        for k, threshold in enumerate(x_t):
            self_training_clf = SelfTrainingClassifier(modelnb, threshold=threshold)
            self_training_clf.fit(X_train, y_train)
            y_predt = self_training_clf.predict(X_test)
            scorest[i] = roc_auc_score(y_test, y_pred)
        max_scoret = np.max(scores)
        max_threshold = x_values[np.argmax(scores)]  
        self_training_clf = SelfTrainingClassifier(modelnb, threshold=max_threshold)
        self_training_clf.fit(X_train, y_train)
        y_pred = self_training_clf.predict(X_test)
        scores[i] = roc_auc_score(y_test, y_pred)
        scoresf[i] = precision_score(y_test, y_pred)
        scoresac[i] = f1_score(y_test, y_pred)
    return (x_values , scores , scoresf , scoresac , auc_scores , F1s ,  F1ss , X_test, y_test)


# In[ ]:


results4 = []
for i in range(10):
    result = f4()
    results4.append(result)
print("Results:", results4)


# In[ ]:


mn_scoresA = np.mean([results4[i][1] for i in range(10)], axis=0)
mn_scoresFB = np.mean([results4[i][2] for i in range(10)], axis=0)
mn_scoresF1 = np.mean([results4[i][3] for i in range(10)], axis=0)
std_score = np.std([results4[i][3] for i in range(10)], axis=0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results4[0][0], results4[0][3], label='Scores1')
ax.plot(results4[0][0], results4[1][3], label='Scores2')
ax.plot(results4[0][0], results4[2][3], label='Scores3')
ax.plot(results4[0][0], results4[3][3], label='Scores4')
ax.plot(results4[0][0], results4[4][3], label='Scores5')
ax.plot(results4[0][0], results4[5][3], label='Scores6')
ax.plot(results4[0][0], results4[6][3], label='Scores7')
ax.plot(results4[0][0], results4[7][3], label='Scores8')
ax.plot(results4[0][0], results4[8][3], label='Scores9')
ax.plot(results4[0][0], results4[9][3], label='Scores10')
ax.fill_between(results4[0][0], mn_scoresF1-std_score, mn_scoresF1+std_score, alpha=0.2)
ax.legend()
ax.set_title('Shuffle Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('Accuracy score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCf = np.mean([i[3] for i in results4], axis=0)
for i in range(len(results4[0][0])):
    mean_SVCsupervised = np.mean([i[6] for i in results4], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[3] for i in results4], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results4[0][0], mean_SVCf, label='self-training F1' , color='red')
ax.fill_between(results4[0][0], mean_SVCf-std_score/2, mean_SVCf+std_score/2, alpha=0.1 , color='red')
ax.plot(results4[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Naive Bayes')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('F1 score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCp = np.mean([i[2] for i in results4], axis=0)
for i in range(len(results4[0][0])):
    mean_SVCsupervised = np.mean([i[5] for i in results4], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[2] for i in results4], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results4[0][0], mean_SVCp, label='self-training F1' , color='blue')
ax.fill_between(results4[0][0], mean_SVCp-std_score/2, mean_SVCp+std_score/2, alpha=0.1 , color='blue')
ax.plot(results4[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Naive Bayes')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('precision score')
plt.show()


# In[ ]:


mean_SVCf = np.mean([i[3] for i in results4], axis=0)
mean_SVCp = np.mean([i[2] for i in results4], axis=0)
std_SVCf = np.std([i[3] for i in results4], axis=0)
std_SVCp = np.std([i[2] for i in results4], axis=0)


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCr4=((mean_SVCf*mean_SVCp)/((2*mean_SVCp)-mean_SVCf))
for i in range(len(results3[0][0])):
    mean_SVCsupervised = np.mean([(i[6]*i[5]/((2*i[5])-i[6])) for i in results4], axis=0)
    z.append(mean_SVCsupervised)
std_score=(std_SVCf*std_SVCp/((2*std_SVCp)-std_SVCf))

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results4[0][0], mean_SVCr4, label='self-training recall' , color='red')
ax.fill_between(results4[0][0], mean_SVCr4-std_score/2, mean_SVCr4+std_score/2, alpha=0.1 , color='red')
ax.plot(results4[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Naive Bayes')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('recall score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVC = np.mean([i[1] for i in results4], axis=0)
for i in range(len(results4[0][0])):
    mean_SVCsupervised = np.mean([i[4] for i in results4], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[1] for i in results4], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results4[0][0], mean_SVC, label='self-training ROC-AUC' , color='brown')
ax.fill_between(results4[0][0], mean_SVC-std_score/2, mean_SVC+std_score/2, alpha=0.1 , color='brown')
ax.plot(results4[0][0], z, label='supervised ROC-AUC ', color='green', linestyle='--')

ax.legend()
ax.set_title('Naive Bayes')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('ROC-AUC score')
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X_train, y_train)
df_train = pd.DataFrame(X_bal)
df_train['Response'] = y_bal
df_train['Random_Mask'] = True
df_train['New_Response']=df_train['Response']
df_train_main=df_train.copy()
df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
df_train_labeled=df_train[df_train['New_Response']!=-1]        
X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
y_train_labeled = df_train_labeled['New_Response']
param_dist =  {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-10, 1e-11, 1e-12, 1e-4]}

model = GaussianNB()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modelnb = GaussianNB(**random_search.best_params_)
modelnb.fit(X_train_labeled, y_train_labeled)
plot_confusion_matrix(modelnb, X_test, y_test, cmap='Blues', normalize='true',
                 display_labels=['Response', 'No Response']);


# In[ ]:


a=[]
b=[]
c=[]
mean_listauc = [] 
mean_listf1 = [] 
mean_listacc = [] 
for i in range(10):
    mean_val1 = (results4[i][1][7])
    mean_val2 = (results4[i][2][7])
    mean_val3 = (results4[i][3][7])
    aa=resultsss[i][9]
    bb=resultsss[i][10]
    cc=resultsss[i][11]
    a.append(aa)
    b.append(bb)
    c.append(cc)   
    mean_listauc.append(mean_val1)
    mean_listf1.append(mean_val3)
    mean_listacc.append(mean_val2)
    
mean_supervisedauc = np.mean(a)
mean_supervisedpre = np.mean(b)
mean_supervisedf1 = np.mean(c)
    
mean_self_trainauc=np.mean(mean_listauc)
mean_self_trainfpre=np.mean(mean_listacc)
mean_self_trainf1=np.mean(mean_listf1)


print("Self-training AUC score with Naive Bayes classifier is:", "{:.3f}%".format(mean_self_trainauc * 100))
print("Supervised Naive Bayes classifier model AUC score is:", "{:.3f}%".format(mean_supervisedauc * 100))
if mean_self_trainauc > mean_supervisedauc:
    print("Self-training AUC score with Naive Bayes classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training AUC score with Naive Bayes classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training precision with Naive Bayes classifier is:", "{:.3f}%".format(mean_self_trainfpre * 100))
print("Supervised Naive Bayes model precision is:", "{:.3f}%".format(mean_supervisedpre * 100))
if mean_self_trainfpre > mean_supervisedpre:
    print("Self-training precision with Naive Bayes classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training precision with Naive Bayes classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training F1 with logistic Naive Bayes is:", "{:.3f}%".format(mean_self_trainf1 * 100))
print("Supervised Naive Bayes model accuracy is:", "{:.3f}%".format(mean_supervisedf1 * 100))
if mean_self_trainf1 > mean_supervisedf1:
    print("Self-training F1 with Naive Bayes classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training F1 with Naive Bayes classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training recall with Naive Bayes is:", "{:.3f}%".format(((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1)))  * 100))
print("Supervised Naive Bayes model recall is:", "{:.3f}%".format(((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))* 100))
if (((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1))))  > (((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))):
    print("Self-training recall with Naive Bayes  is greater than supervised model for 24% of labeled data")
else:
    print("Self-training recall with Naive Bayes  is not greater than supervised model for 24% of labeled data")


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB, mn_scoresF1,mn_scoresA)]
mean_SVC = np.mean([i[1] for i in results4], axis=0)
    
max_index = np.argmax(w[0][1]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][1][max_index]
print("maximum  precision of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum precision:", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][2]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][2][max_index]
print("maximum F1 of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum F1 :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][3]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][3][max_index]
print("maximum AUC  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum AUC :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax((mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = (mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))[max_index]
print("maximum recall  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum recall :", "{:.1f}%".format(max_value_first_row*100))


# In[ ]:


df = pd.read_csv("C:/Users/gebruiker/Desktop/New folder/marketing_campaign.csv", sep='\t')
df["Income"] = df.groupby("Education")['Income'].transform(lambda x: x.fillna(x.mean()))
df.drop(['Z_CostContact', 'Z_Revenue' , "ID"], axis=1, inplace=True)
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
df['Age'] = df.Year_Birth.apply(lambda x: 2023 - int(x))
df.drop(columns=["Year_Birth"],axis=1 , inplace=True)
df['Education'].replace(['PhD', 'Master','2n Cycle'],'PostGraduate', inplace=True)  
df['Education'].replace(['Graduation'], 'Graduate', inplace=True)
df['Education'].replace(['Basic'],'UnderGraduate', inplace=True)
df["Children"]=df["Kidhome"]+df["Teenhome"]
df.drop(columns=["Kidhome" , "Teenhome"],axis=1 , inplace=True)
categorical_data = df[['Education', 'Marital_Status']]
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
df=df.rename(columns={"NumDealsPurchases": "Deals","NumWebPurchases":"Web","NumCatalogPurchases":"Catalog","NumStorePurchases":"Store","NumWebVisitsMonth":"WebVisits"})
df=df.rename(columns={"AcceptedCmp1": "A1","AcceptedCmp2":"A2","AcceptedCmp3":"A3","AcceptedCmp4":"A4","AcceptedCmp5":"A5"})
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = df['Dt_Customer'].max()
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Time_Enrolled_Days'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df=df.drop(columns=['Dt_Customer','Date_Collected'],axis=1)
df['ActiveDays'] = df['Time_Enrolled_Days'] - df['Recency']
df["Family_Size"] = df["Marital_Status"].replace({"Single": 1, "Couple":2}) + df["Children"]
df.drop(df[(df['Income'] > (df['Income'].mean() * 2)) | (df['Age'] > (df['Age'].mean() * 2))].index, inplace=True)
df['AM.Total'] = df['Wines']+df['Fruits']+df['Meat']+df['Fish']+df['Sweets']+df['Gold']
df['AllPurchase'] = df['Web']+df['Catalog']+df['Store']
df['Average.P'] = round((df['AM.Total'] / df['AllPurchase']), 1)
df['ShareDealsPurchase'] = round((df['Deals'] / df['AllPurchase']) * 100, 1)
df['TotalAcceptedCmp'] = df['A1']+df['A2']+df['A3']+df['A4']+df['A5']
df=df.drop(columns=['A1','A2','A3','A4','A5'],axis=1)
df = df[df["AllPurchase"] != 0]
df.drop(df[df['Average.P'] > (df['Average.P'].mean() * 4)].index, inplace=True)
df.drop(columns=["Average.P"],axis=1 , inplace=True)
X= df.drop('Response', axis=1).values
y =df['Response'].values
ct = ColumnTransformer([
    ('catagoric1', OneHotEncoder(), [0]),
    ('catagoric2', OneHotEncoder(), [1]),
    ('numeric', StandardScaler(), list(range(2, len(X.T))))])
scaled_ds = ct.fit_transform(X)
print('Preprocessed Data:')
print(scaled_ds[0])
print(scaled_ds.shape)


# In[ ]:


def f6():
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    param_dist = {'n_neighbors': [5, 7, 9, 10], 'leaf_size': [20,30,50],
             'metric': ['euclidean', 'manhattan' , 'minkowski']}

    model = KNeighborsClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist)
    random_search.fit(X_train_labeled, y_train_labeled)
    modelkn = KNeighborsClassifier(**random_search.best_params_)
    modelkn.fit(X_train_labeled, y_train_labeled)
    y_pred = modelkn.predict(X_test)
    auc_scores = roc_auc_score(y_test, y_pred)
    F1s = precision_score(y_test, y_pred)
    F1ss = f1_score(y_test, y_pred)
          
    x_values = np.arange(0.03, 0.50, 0.03)
    x_values = np.append(x_values, 0.49)
    scores = np.empty((x_values.shape[0],))
    scoresf =np.empty((x_values.shape[0],))
    scoresac =np.empty((x_values.shape[0],))
    for i, j in enumerate(x_values):
        df_train_main.loc[df_train_main.sample(frac=j).index, 'Random_Mask'] = False
        df_train_main['New_Response']= df_train_main.apply(lambda x:x['Response']if x['Random_Mask'] == False else -1,axis=1)
        X_train = df_train_main.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
        df_train_labeled=df_train[df_train['New_Response']!=-1]
        param_dist =   {'n_neighbors': [5, 7, 9, 10], 'leaf_size': [20,30,50],
             'metric': ['euclidean', 'manhattan' , 'minkowski']}
        model = KNeighborsClassifier()
        random_search = RandomizedSearchCV(estimator=model,
        param_distributions=param_dist)
        random_search.fit(X_train_labeled, y_train_labeled)        
        modelkn = KNeighborsClassifier(**random_search.best_params_ )       
        y_train = df_train_main['New_Response']
        
        x_t = np.arange(0.4, 1.05, 0.05)
        x_t = np.append(x_values, 0.99999)
        scorest = np.empty((x_values.shape[0],))
        for k, threshold in enumerate(x_t):
            self_training_clf = SelfTrainingClassifier(modelkn, threshold=threshold)
            self_training_clf.fit(X_train, y_train)
            y_predt = self_training_clf.predict(X_test)
            scorest[i] = roc_auc_score(y_test, y_pred)
        max_scoret = np.max(scores)
        max_threshold = x_values[np.argmax(scores)]  
        self_training_clf = SelfTrainingClassifier(modelkn, threshold=max_threshold)
        self_training_clf.fit(X_train, y_train)
        y_pred = self_training_clf.predict(X_test)
        scores[i] = roc_auc_score(y_test, y_pred)
        scoresf[i] = precision_score(y_test, y_pred)
        scoresac[i] = f1_score(y_test, y_pred)
    return (x_values , scores , scoresf , scoresac , auc_scores , F1s ,  F1ss , X_test, y_test)


# In[ ]:


results6 = []
for i in range(10):
    result = f6()
    results6.append(result)
print("Results:", results6)


# In[ ]:


mn_scoresA = np.mean([results6[i][1] for i in range(10)], axis=0)
mn_scoresFB = np.mean([results6[i][2] for i in range(10)], axis=0)
mn_scoresF1 = np.mean([results6[i][3] for i in range(10)], axis=0)
std_score = np.std([results6[i][3] for i in range(10)], axis=0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results6[0][0], results6[0][3], label='Scores1')
ax.plot(results6[0][0], results6[1][3], label='Scores2')
ax.plot(results6[0][0], results6[2][3], label='Scores3')
ax.plot(results6[0][0], results6[3][3], label='Scores4')
ax.plot(results6[0][0], results6[4][3], label='Scores5')
ax.plot(results6[0][0], results6[5][3], label='Scores6')
ax.plot(results6[0][0], results6[6][3], label='Scores7')
ax.plot(results6[0][0], results6[7][3], label='Scores8')
ax.plot(results6[0][0], results6[8][3], label='Scores9')
ax.plot(results6[0][0], results6[9][3], label='Scores10')
ax.fill_between(results6[0][0], mn_scoresF1-std_score, mn_scoresF1+std_score, alpha=0.2)
ax.legend()
ax.set_title('Shuffle Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('Accuracy score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCf = np.mean([i[3] for i in results6], axis=0)
for i in range(len(results6[0][0])):
    mean_SVCsupervised = np.mean([i[6] for i in results6], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[3] for i in results6], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results6[0][0], mean_SVCf, label='self-training F1' , color='red')
ax.fill_between(results6[0][0], mean_SVCf-std_score/2, mean_SVCf+std_score/2, alpha=0.1 , color='red')
ax.plot(results6[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('KNN')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('F1 score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCp = np.mean([i[2] for i in results6], axis=0)
for i in range(len(results6[0][0])):
    mean_SVCsupervised = np.mean([i[5] for i in results6], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[2] for i in results6], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results6[0][0], mean_SVCp, label='self-training F1' , color='blue')
ax.fill_between(results6[0][0], mean_SVCp-std_score/2, mean_SVCp+std_score/2, alpha=0.1 , color='blue')
ax.plot(results6[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('KNN')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('precision score')
plt.show()


# In[ ]:


mean_SVCf = np.mean([i[3] for i in results6], axis=0)
mean_SVCp = np.mean([i[2] for i in results6], axis=0)
std_SVCf = np.std([i[3] for i in results6], axis=0)
std_SVCp = np.std([i[2] for i in results6], axis=0)

w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCr6=(mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))
for i in range(len(results6[0][0])):
    mean_SVCsupervised = np.mean([(i[6]*i[5]/((2*i[5])-i[6])) for i in results6], axis=0)
    z.append(mean_SVCsupervised)
std_score=(std_SVCf*std_SVCp/((2*std_SVCp)-std_SVCf))

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results6[0][0], mean_SVCr6, label='self-training recall' , color='red')
ax.fill_between(results6[0][0], mean_SVCr6-std_score/2, mean_SVCr6+std_score/2, alpha=0.1 , color='red')
ax.plot(results6[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('KNN')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('recall score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVC = np.mean([i[1] for i in results6], axis=0)
for i in range(len(results6[0][0])):
    mean_SVCsupervised = np.mean([i[4] for i in results6], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[1] for i in results6], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results6[0][0], mean_SVC, label='self-training F1' , color='brown')
ax.fill_between(results6[0][0], mean_SVC-std_score/2, mean_SVC+std_score/2, alpha=0.1 , color='brown')
ax.plot(results6[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('KNN')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('ROC-AUC score')
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X_train, y_train)
df_train = pd.DataFrame(X_bal)
df_train['Response'] = y_bal
df_train['Random_Mask'] = True
df_train['New_Response']=df_train['Response']
df_train_main=df_train.copy()
df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
df_train_labeled=df_train[df_train['New_Response']!=-1]        
X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
y_train_labeled = df_train_labeled['New_Response']
param_dist = {'n_neighbors': [5, 7, 9, 10], 'leaf_size': [20,30,50],
         'metric': ['euclidean', 'manhattan' , 'minkowski']}

model = KNeighborsClassifier()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modelkn = KNeighborsClassifier(**random_search.best_params_)
modelkn.fit(X_train_labeled, y_train_labeled)
plot_confusion_matrix(modelkn, X_test, y_test, cmap='Blues', normalize='true',
                 display_labels=['Response', 'No Response']);


# In[ ]:


a=[]
b=[]
c=[]
mean_listauc = [] 
mean_listf1 = [] 
mean_listacc = [] 
for i in range(10):
    mean_val1 = (results6[i][1][7])
    mean_val2 = (results6[i][2][7])
    mean_val3 = (results6[i][3][7])
    aa=resultsss[i][12]
    bb=resultsss[i][13]
    cc=resultsss[i][14]
    a.append(aa)
    b.append(bb)
    c.append(cc)   
    mean_listauc.append(mean_val1)
    mean_listf1.append(mean_val3)
    mean_listacc.append(mean_val2)
    
mean_supervisedauc = np.mean(a)
mean_supervisedpre = np.mean(b)
mean_supervisedf1 = np.mean(c)
    
mean_self_trainauc=np.mean(mean_listauc)
mean_self_trainfpre=np.mean(mean_listacc)
mean_self_trainf1=np.mean(mean_listf1)


print("Self-training AUC score with KNN classifier is:", "{:.3f}%".format(mean_self_trainauc * 100))
print("Supervised KNN classifier model AUC score is:", "{:.3f}%".format(mean_supervisedauc * 100))
if mean_self_trainauc > mean_supervisedauc:
    print("Self-training AUC score with KNN classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training AUC score with KNN classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training precision with KNN classifier is:", "{:.3f}%".format(mean_self_trainfpre * 100))
print("Supervised KNN model precision is:", "{:.3f}%".format(mean_supervisedpre * 100))
if mean_self_trainfpre > mean_supervisedpre:
    print("Self-training precision with KNN classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training precision with KNN classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training F1 with logistic KNN is:", "{:.3f}%".format(mean_self_trainf1 * 100))
print("Supervised KNN model accuracy is:", "{:.3f}%".format(mean_supervisedf1 * 100))
if mean_self_trainf1 > mean_supervisedf1:
    print("Self-training F1 with KNN classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training F1 with KNN classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training recall with KNN is:", "{:.3f}%".format(((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1)))  * 100))
print("Supervised KNN model recall is:", "{:.3f}%".format(((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))* 100))
if (((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1))))  > (((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))):
    print("Self-training recall with KNN  is greater than supervised model for 24% of labeled data")
else:
    print("Self-training recall with KNN  is not greater than supervised model for 24% of labeled data")


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB, mn_scoresF1,mn_scoresA)]
mean_SVC = np.mean([i[1] for i in results6], axis=0)
    
max_index = np.argmax(w[0][1]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][1][max_index]
print("maximum  precision of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum precision:", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][2]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][2][max_index]
print("maximum F1 of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum F1 :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][3]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][3][max_index]
print("maximum AUC  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum AUC :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax((mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = (mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))[max_index]
print("maximum recall  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum recall :", "{:.1f}%".format(max_value_first_row*100))


# In[ ]:


df = pd.read_csv("C:/Users/gebruiker/Desktop/New folder/marketing_campaign.csv", sep='\t')
df["Income"] = df.groupby("Education")['Income'].transform(lambda x: x.fillna(x.mean()))
df.drop(['Z_CostContact', 'Z_Revenue' , "ID"], axis=1, inplace=True)
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
df['Age'] = df.Year_Birth.apply(lambda x: 2023 - int(x))
df.drop(columns=["Year_Birth"],axis=1 , inplace=True)
df['Education'].replace(['PhD', 'Master','2n Cycle'],'PostGraduate', inplace=True)  
df['Education'].replace(['Graduation'], 'Graduate', inplace=True)
df['Education'].replace(['Basic'],'UnderGraduate', inplace=True)
df["Children"]=df["Kidhome"]+df["Teenhome"]
df.drop(columns=["Kidhome" , "Teenhome"],axis=1 , inplace=True)
categorical_data = df[['Education', 'Marital_Status']]
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
df=df.rename(columns={"NumDealsPurchases": "Deals","NumWebPurchases":"Web","NumCatalogPurchases":"Catalog","NumStorePurchases":"Store","NumWebVisitsMonth":"WebVisits"})
df=df.rename(columns={"AcceptedCmp1": "A1","AcceptedCmp2":"A2","AcceptedCmp3":"A3","AcceptedCmp4":"A4","AcceptedCmp5":"A5"})
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = df['Dt_Customer'].max()
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Time_Enrolled_Days'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df=df.drop(columns=['Dt_Customer','Date_Collected'],axis=1)
df['ActiveDays'] = df['Time_Enrolled_Days'] - df['Recency']
df["Family_Size"] = df["Marital_Status"].replace({"Single": 1, "Couple":2}) + df["Children"]
df.drop(df[(df['Income'] > (df['Income'].mean() * 2)) | (df['Age'] > (df['Age'].mean() * 2))].index, inplace=True)
df['AM.Total'] = df['Wines']+df['Fruits']+df['Meat']+df['Fish']+df['Sweets']+df['Gold']
df['AllPurchase'] = df['Web']+df['Catalog']+df['Store']
df['Average.P'] = round((df['AM.Total'] / df['AllPurchase']), 1)
df['ShareDealsPurchase'] = round((df['Deals'] / df['AllPurchase']) * 100, 1)
df['TotalAcceptedCmp'] = df['A1']+df['A2']+df['A3']+df['A4']+df['A5']
df=df.drop(columns=['A1','A2','A3','A4','A5'],axis=1)
df = df[df["AllPurchase"] != 0]
df.drop(df[df['Average.P'] > (df['Average.P'].mean() * 4)].index, inplace=True)
df.drop(columns=["Average.P"],axis=1 , inplace=True)
X= df.drop('Response', axis=1).values
y =df['Response'].values
ct = ColumnTransformer([
    ('catagoric1', OneHotEncoder(), [0]),
    ('catagoric2', OneHotEncoder(), [1]),
    ('numeric', StandardScaler(), list(range(2, len(X.T))))])
scaled_ds = ct.fit_transform(X)
print('Preprocessed Data:')
print(scaled_ds[0])
print(scaled_ds.shape)


# In[ ]:


def f7():
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    param_dist = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [1 ,2, 3, 4, 5, 6, 7, 8],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8]}

    model = DecisionTreeClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist)
    random_search.fit(X_train_labeled, y_train_labeled)
    modelDT = DecisionTreeClassifier(**random_search.best_params_)
    modelDT.fit(X_train_labeled, y_train_labeled)
    y_pred = modelDT.predict(X_test)
    auc_scores = roc_auc_score(y_test, y_pred)
    F1s = precision_score(y_test, y_pred)
    F1ss = f1_score(y_test, y_pred)
          
    x_values = np.arange(0.03, 0.50, 0.03)
    x_values = np.append(x_values, 0.49)
    scores = np.empty((x_values.shape[0],))
    scoresf =np.empty((x_values.shape[0],))
    scoresac =np.empty((x_values.shape[0],))
    for i, j in enumerate(x_values):
        df_train_main.loc[df_train_main.sample(frac=j).index, 'Random_Mask'] = False
        df_train_main['New_Response']= df_train_main.apply(lambda x:x['Response']if x['Random_Mask'] == False else -1,axis=1)
        X_train = df_train_main.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
        df_train_labeled=df_train[df_train['New_Response']!=-1]
        param_dist = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [1 ,2, 3, 4, 5, 6, 7, 8],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8]}

        model = DecisionTreeClassifier()
        random_search = RandomizedSearchCV(estimator=model,
        param_distributions=param_dist)
        random_search.fit(X_train_labeled, y_train_labeled)        
        modelDT = DecisionTreeClassifier(**random_search.best_params_ )       
        y_train = df_train_main['New_Response']
        
        x_t = np.arange(0.4, 1.05, 0.05)
        x_t = np.append(x_values, 0.99999)
        scorest = np.empty((x_values.shape[0],))
        for k, threshold in enumerate(x_t):
            self_training_clf = SelfTrainingClassifier(modelDT, threshold=threshold)
            self_training_clf.fit(X_train, y_train)
            y_predt = self_training_clf.predict(X_test)
            scorest[i] = roc_auc_score(y_test, y_pred)
        max_scoret = np.max(scores)
        max_threshold = x_values[np.argmax(scores)]  
        self_training_clf = SelfTrainingClassifier(modelDT, threshold=max_threshold)
        self_training_clf.fit(X_train, y_train)
        y_pred = self_training_clf.predict(X_test)
        scores[i] = roc_auc_score(y_test, y_pred)
        scoresf[i] = precision_score(y_test, y_pred)
        scoresac[i] = f1_score(y_test, y_pred)
    return (x_values , scores , scoresf , scoresac , auc_scores , F1s ,  F1ss , X_test, y_test)


# In[ ]:


results7 = []
for i in range(10):
    result = f7()
    results7.append(result)
print("Results:", results7)


# In[ ]:


mn_scoresA = np.mean([results77[i][1] for i in range(10)], axis=0)
mn_scoresFB = np.mean([results77[i][2] for i in range(10)], axis=0)
mn_scoresF1 = np.mean([results77[i][3] for i in range(10)], axis=0)
std_score = np.std([results77[i][3] for i in range(10)], axis=0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results77[0][0], results77[0][3], label='Scores1')
ax.plot(results77[0][0], results77[1][3], label='Scores2')
ax.plot(results77[0][0], results77[2][3], label='Scores3')
ax.plot(results77[0][0], results77[3][3], label='Scores4')
ax.plot(results77[0][0], results77[4][3], label='Scores5')
ax.plot(results77[0][0], results77[5][3], label='Scores6')
ax.plot(results77[0][0], results77[6][3], label='Scores7')
ax.plot(results77[0][0], results77[7][3], label='Scores8')
ax.plot(results77[0][0], results77[8][3], label='Scores9')
ax.plot(results77[0][0], results77[9][3], label='Scores10')
ax.fill_between(results77[0][0], mn_scoresF1-std_score, mn_scoresF1+std_score, alpha=0.2)
ax.legend()
ax.set_title('Shuffle Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('Accuracy score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCf = np.mean([i[3] for i in results77], axis=0)
for i in range(len(results77[0][0])):
    mean_SVCsupervised = np.mean([i[6] for i in results77], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[3] for i in results77], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results77[0][0], mean_SVCf, label='self-training F1' , color='red')
ax.fill_between(results77[0][0], mean_SVCf-std_score/2, mean_SVCf+std_score/2, alpha=0.1 , color='red')
ax.plot(results77[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('DecisionTree')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('F1 score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCp = np.mean([i[2] for i in results77], axis=0)
for i in range(len(results77[0][0])):
    mean_SVCsupervised = np.mean([i[5] for i in results77], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[2] for i in results77], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results77[0][0], mean_SVCp, label='self-training F1' , color='blue')
ax.fill_between(results77[0][0], mean_SVCp-std_score/2, mean_SVCp+std_score/2, alpha=0.1 , color='blue')
ax.plot(results77[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('DecisionTree')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('precission score')
plt.show()


# In[ ]:


mean_SVCf = np.mean([i[3] for i in results77], axis=0)
mean_SVCp = np.mean([i[2] for i in results77], axis=0)
std_SVCf = np.std([i[3] for i in results77], axis=0)
std_SVCp = np.std([i[2] for i in results77], axis=0)

w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCr7=(mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))
for i in range(len(results77[0][0])):
    mean_SVCsupervised = np.mean([(i[6]*i[5]/((2*i[5])-i[6])) for i in results77], axis=0)
    z.append(mean_SVCsupervised)
std_score=(std_SVCf*std_SVCp/((2*std_SVCp)-std_SVCf))

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results77[0][0], mean_SVCr7, label='self-training recall' , color='red')
ax.fill_between(results77[0][0], mean_SVCr7-std_score/2, mean_SVCr7+std_score/2, alpha=0.1 , color='red')
ax.plot(results77[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('DecisionTree')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('recall score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVC = np.mean([i[1] for i in results77], axis=0)
for i in range(len(results77[0][0])):
    mean_SVCsupervised = np.mean([i[4] for i in results77], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[1] for i in results77], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results77[0][0], mean_SVC, label='self-training F1' , color='brown')
ax.fill_between(results77[0][0], mean_SVC-std_score/2, mean_SVC+std_score/2, alpha=0.1 , color='brown')
ax.plot(results77[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('DecisionTree')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('ROC-AUC score')
plt.show()


# In[ ]:


a=[]
b=[]
c=[]
mean_listauc = [] 
mean_listf1 = [] 
mean_listacc = [] 
for i in range(10):
    mean_val1 = (results77[i][1][7])
    mean_val2 = (results77[i][2][7])
    mean_val3 = (results77[i][3][7])
    aa=resultsss[i][15]
    bb=resultsss[i][16]
    cc=resultsss[i][17]
    a.append(aa)
    b.append(bb)
    c.append(cc)   
    mean_listauc.append(mean_val1)
    mean_listf1.append(mean_val3)
    mean_listacc.append(mean_val2)
    
mean_supervisedauc = np.mean(a)
mean_supervisedpre = np.mean(b)
mean_supervisedf1 = np.mean(c)
    
mean_self_trainauc=np.mean(mean_listauc)
mean_self_trainfpre=np.mean(mean_listacc)
mean_self_trainf1=np.mean(mean_listf1)


print("Self-training AUC score with Decissiontree classifier is:", "{:.3f}%".format(mean_self_trainauc * 100))
print("Supervised Decissiontree classifier model AUC score is:", "{:.3f}%".format(mean_supervisedauc * 100))
if mean_self_trainauc > mean_supervisedauc:
    print("Self-training AUC score with Decissiontree classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training AUC score with Decissiontree classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training precision with Decissiontree classifier is:", "{:.3f}%".format(mean_self_trainfpre * 100))
print("Supervised Decissiontree model precision is:", "{:.3f}%".format(mean_supervisedpre * 100))
if mean_self_trainfpre > mean_supervisedpre:
    print("Self-training precision with Decissiontree classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training precision with Decissiontree classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training F1 with logistic Decissiontree is:", "{:.3f}%".format(mean_self_trainf1 * 100))
print("Supervised KNN model accuracy is:", "{:.3f}%".format(mean_supervisedf1 * 100))
if mean_self_trainf1 > mean_supervisedf1:
    print("Self-training F1 with Decissiontree classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training F1 with Decissiontree classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training recall with Decissiontree is:", "{:.3f}%".format(((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1)))  * 100))
print("Supervised Decissiontree model recall is:", "{:.3f}%".format(((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))* 100))
if (((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1))))  > (((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))):
    print("Self-training recall with Decissiontree  is greater than supervised model for 24% of labeled data")
else:
    print("Self-training recall with Decissiontree  is not greater than supervised model for 24% of labeled data")


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB, mn_scoresF1,mn_scoresA)]
mean_SVC = np.mean([i[1] for i in results77], axis=0)
    
max_index = np.argmax(w[0][1]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][1][max_index]
print("maximum  precision of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum precision:", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][2]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][2][max_index]
print("maximum F1 of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum F1 :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][3]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][3][max_index]
print("maximum AUC  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum AUC :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax((mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = (mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))[max_index]
print("maximum recall  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum recall :", "{:.1f}%".format(max_value_first_row*100))


# In[ ]:


df = pd.read_csv("C:/Users/gebruiker/Desktop/New folder/marketing_campaign.csv", sep='\t')
df["Income"] = df.groupby("Education")['Income'].transform(lambda x: x.fillna(x.mean()))
df.drop(['Z_CostContact', 'Z_Revenue' , "ID"], axis=1, inplace=True)
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
df['Age'] = df.Year_Birth.apply(lambda x: 2023 - int(x))
df.drop(columns=["Year_Birth"],axis=1 , inplace=True)
df['Education'].replace(['PhD', 'Master','2n Cycle'],'PostGraduate', inplace=True)  
df['Education'].replace(['Graduation'], 'Graduate', inplace=True)
df['Education'].replace(['Basic'],'UnderGraduate', inplace=True)
df["Children"]=df["Kidhome"]+df["Teenhome"]
df.drop(columns=["Kidhome" , "Teenhome"],axis=1 , inplace=True)
categorical_data = df[['Education', 'Marital_Status']]
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
df=df.rename(columns={"NumDealsPurchases": "Deals","NumWebPurchases":"Web","NumCatalogPurchases":"Catalog","NumStorePurchases":"Store","NumWebVisitsMonth":"WebVisits"})
df=df.rename(columns={"AcceptedCmp1": "A1","AcceptedCmp2":"A2","AcceptedCmp3":"A3","AcceptedCmp4":"A4","AcceptedCmp5":"A5"})
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = df['Dt_Customer'].max()
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Time_Enrolled_Days'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df=df.drop(columns=['Dt_Customer','Date_Collected'],axis=1)
df['ActiveDays'] = df['Time_Enrolled_Days'] - df['Recency']
df["Family_Size"] = df["Marital_Status"].replace({"Single": 1, "Couple":2}) + df["Children"]
df.drop(df[(df['Income'] > (df['Income'].mean() * 2)) | (df['Age'] > (df['Age'].mean() * 2))].index, inplace=True)
df['AM.Total'] = df['Wines']+df['Fruits']+df['Meat']+df['Fish']+df['Sweets']+df['Gold']
df['AllPurchase'] = df['Web']+df['Catalog']+df['Store']
df['Average.P'] = round((df['AM.Total'] / df['AllPurchase']), 1)
df['ShareDealsPurchase'] = round((df['Deals'] / df['AllPurchase']) * 100, 1)
df['TotalAcceptedCmp'] = df['A1']+df['A2']+df['A3']+df['A4']+df['A5']
df=df.drop(columns=['A1','A2','A3','A4','A5'],axis=1)
df = df[df["AllPurchase"] != 0]
df.drop(df[df['Average.P'] > (df['Average.P'].mean() * 4)].index, inplace=True)
df.drop(columns=["Average.P"],axis=1 , inplace=True)
X= df.drop('Response', axis=1).values
y =df['Response'].values
ct = ColumnTransformer([
    ('catagoric1', OneHotEncoder(), [0]),
    ('catagoric2', OneHotEncoder(), [1]),
    ('numeric', StandardScaler(), list(range(2, len(X.T))))])
scaled_ds = ct.fit_transform(X)
print('Preprocessed Data:')
print(scaled_ds[0])
print(scaled_ds.shape)


# In[ ]:


def f8():
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    param_dist =  {
    'n_estimators': sp_randint(50,100, 200),
    'learning_rate': [0.1, 0.01, 0.001 , 1],
    'max_depth': sp_randint(2, 10),
    'subsample': [0.5, 2, 1.0]}

    model = GradientBoostingClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist)
    random_search.fit(X_train_labeled, y_train_labeled)
    modelgb = GradientBoostingClassifier(**random_search.best_params_)
    modelgb.fit(X_train_labeled, y_train_labeled)
    y_pred = modelgb.predict(X_test)
    auc_scores = roc_auc_score(y_test, y_pred)
    F1s = precision_score(y_test, y_pred)
    F1ss = f1_score(y_test, y_pred)
          
    x_values = np.arange(0.03, 0.50, 0.03)
    x_values = np.append(x_values, 0.49)
    scores = np.empty((x_values.shape[0],))
    scoresf =np.empty((x_values.shape[0],))
    scoresac =np.empty((x_values.shape[0],))
    for i, j in enumerate(x_values):
        df_train_main.loc[df_train_main.sample(frac=j).index, 'Random_Mask'] = False
        df_train_main['New_Response']= df_train_main.apply(lambda x:x['Response']if x['Random_Mask'] == False else -1,axis=1)
        X_train = df_train_main.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
        df_train_labeled=df_train[df_train['New_Response']!=-1]
        param_dist =  {
        'n_estimators': sp_randint(50,100, 200),
        'learning_rate': [0.1, 0.01, 0.001 , 1],
        'max_depth': sp_randint(2, 10),
        'subsample': [0.5, 2, 1.0]}

        model = GradientBoostingClassifier()
        random_search = RandomizedSearchCV(estimator=model,
        param_distributions=param_dist)
        random_search.fit(X_train_labeled, y_train_labeled)        
        modelgb = GradientBoostingClassifier(**random_search.best_params_ )       
        y_train = df_train_main['New_Response']
        
        x_t = np.arange(0.4, 1.05, 0.05)
        x_t = np.append(x_values, 0.99999)
        scorest = np.empty((x_values.shape[0],))
        for k, threshold in enumerate(x_t):
            self_training_clf = SelfTrainingClassifier(modelgb, threshold=threshold)
            self_training_clf.fit(X_train, y_train)
            y_predt = self_training_clf.predict(X_test)
            scorest[i] = roc_auc_score(y_test, y_pred)
        max_scoret = np.max(scores)
        max_threshold = x_values[np.argmax(scores)]  
        self_training_clf = SelfTrainingClassifier(modelgb, threshold=max_threshold)
        self_training_clf.fit(X_train, y_train)
        y_pred = self_training_clf.predict(X_test)
        scores[i] = roc_auc_score(y_test, y_pred)
        scoresf[i] = precision_score(y_test, y_pred)
        scoresac[i] = f1_score(y_test, y_pred)
    return (x_values , scores , scoresf , scoresac , auc_scores , F1s ,  F1ss , X_test, y_test)


# In[ ]:


results8 = []
for i in range(10):
    result = f8()
    results8.append(result)
print("Results:", results8)


# In[ ]:


mn_scoresA = np.mean([results8[i][1] for i in range(10)], axis=0)
mn_scoresFB = np.mean([results8[i][2] for i in range(10)], axis=0)
mn_scoresF1 = np.mean([results8[i][3] for i in range(10)], axis=0)
std_score = np.std([results77[i][3] for i in range(10)], axis=0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results8[0][0], results8[0][3], label='Scores1')
ax.plot(results8[0][0], results8[1][3], label='Scores2')
ax.plot(results8[0][0], results8[2][3], label='Scores3')
ax.plot(results8[0][0], results8[3][3], label='Scores4')
ax.plot(results8[0][0], results8[4][3], label='Scores5')
ax.plot(results8[0][0], results8[5][3], label='Scores6')
ax.plot(results8[0][0], results8[6][3], label='Scores7')
ax.plot(results8[0][0], results8[7][3], label='Scores8')
ax.plot(results8[0][0], results8[8][3], label='Scores9')
ax.plot(results8[0][0], results8[9][3], label='Scores10')
ax.fill_between(results8[0][0], mn_scoresF1-std_score, mn_scoresF1+std_score, alpha=0.2)
ax.legend()
ax.set_title('Shuffle Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('Accuracy score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCf = np.mean([i[3] for i in results8], axis=0)
for i in range(len(results8[0][0])):
    mean_SVCsupervised = np.mean([i[6] for i in results8], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[3] for i in results8], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results8[0][0], mean_SVCf, label='self-training F1' , color='red')
ax.fill_between(results8[0][0], mean_SVCf-std_score/2, mean_SVCf+std_score/2, alpha=0.1 , color='red')
ax.plot(results8[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Gradient Boosting')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('F1 score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB)]
z=[]
mean_SVCp = np.mean([i[2] for i in results8], axis=0)
for i in range(len(results8[0][0])):
    mean_SVCsupervised = np.mean([i[5] for i in results8], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[2] for i in results8], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results8[0][0], mean_SVCp, label='self-training F1' , color='blue')
ax.fill_between(results8[0][0], mean_SVCp-std_score/2, mean_SVCp+std_score/2, alpha=0.1 , color='blue')
ax.plot(results8[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Gradient Boosting')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('Precision')
plt.show()


# In[ ]:


mean_SVCf = np.mean([i[3] for i in results8], axis=0)
mean_SVCp = np.mean([i[2] for i in results8], axis=0)
std_SVCf = np.std([i[3] for i in results8], axis=0)
std_SVCp = np.std([i[2] for i in results8], axis=0)

w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCr8=(mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))
for i in range(len(results8[0][0])):
    mean_SVCsupervised = np.mean([(i[6]*i[5]/((2*i[5])-i[6])) for i in results8], axis=0)
    z.append(mean_SVCsupervised)
std_score=(std_SVCf*std_SVCp/((2*std_SVCp)-std_SVCf))

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results8[0][0], mean_SVCr8, label='self-training recall' , color='red')
ax.fill_between(results8[0][0], mean_SVCr8-std_score/2, mean_SVCr8+std_score/2, alpha=0.1 , color='red')
ax.plot(results8[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Gradient Boosting')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('recall score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresA)]
z=[]
mean_SVC = np.mean([i[1] for i in results8], axis=0)
for i in range(len(results8[0][0])):
    mean_SVCsupervised = np.mean([i[4] for i in results8], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[1] for i in results8], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results8[0][0], mean_SVC, label='self-training F1' , color='brown')
ax.fill_between(results8[0][0], mean_SVC-std_score/2, mean_SVC+std_score/2, alpha=0.1 , color='brown')
ax.plot(results8[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('Gradient Boosting')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('ROC-AUC score')
plt.show()


# In[ ]:


a=[]
b=[]
c=[]
mean_listauc = [] 
mean_listf1 = [] 
mean_listacc = [] 
for i in range(10):
    mean_val1 = (results8[i][1][7])
    mean_val2 = (results8[i][2][7])
    mean_val3 = (results8[i][3][7])
    aa=resultsss[i][18]
    bb=resultsss[i][19]
    cc=resultsss[i][20]
    a.append(aa)
    b.append(bb)
    c.append(cc)   
    mean_listauc.append(mean_val1)
    mean_listf1.append(mean_val3)
    mean_listacc.append(mean_val2)
    
mean_supervisedauc = np.mean(a)
mean_supervisedpre = np.mean(b)
mean_supervisedf1 = np.mean(c)
    
mean_self_trainauc=np.mean(mean_listauc)
mean_self_trainfpre=np.mean(mean_listacc)
mean_self_trainf1=np.mean(mean_listf1)


print("Self-training AUC score with GB classifier is:", "{:.3f}%".format(mean_self_trainauc * 100))
print("Supervised GB classifier model AUC score is:", "{:.3f}%".format(mean_supervisedauc * 100))
if mean_self_trainauc > mean_supervisedauc:
    print("Self-training AUC score with GB classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training AUC score with GB classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training precision with GB classifier is:", "{:.3f}%".format(mean_self_trainfpre * 100))
print("Supervised GB model precision is:", "{:.3f}%".format(mean_supervisedpre * 100))
if mean_self_trainfpre > mean_supervisedpre:
    print("Self-training precision with GB classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training precision with GB classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training F1 with logistic GB is:", "{:.3f}%".format(mean_self_trainf1 * 100))
print("Supervised GB model accuracy is:", "{:.3f}%".format(mean_supervisedf1 * 100))
if mean_self_trainf1 > mean_supervisedf1:
    print("Self-training F1 with GB classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training F1 with GB classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training recall with GB is:", "{:.3f}%".format(((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1)))  * 100))
print("Supervised GB model recall is:", "{:.3f}%".format(((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))* 100))
if (((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1))))  > (((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))):
    print("Self-training recall with GB  is greater than supervised model for 24% of labeled data")
else:
    print("Self-training recall with GB  is not greater than supervised model for 24% of labeled data")


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB, mn_scoresF1,mn_scoresA)]
mean_SVC = np.mean([i[1] for i in results8], axis=0)
    
max_index = np.argmax(w[0][1]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][1][max_index]
print("maximum  precision of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum precision:", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][2]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][2][max_index]
print("maximum F1 of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum F1 :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][3]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][3][max_index]
print("maximum AUC  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum AUC :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax((mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = (mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))[max_index]
print("maximum recall  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum recall :", "{:.1f}%".format(max_value_first_row*100))


# In[ ]:


df = pd.read_csv("C:/Users/gebruiker/Desktop/New folder/marketing_campaign.csv", sep='\t')
df["Income"] = df.groupby("Education")['Income'].transform(lambda x: x.fillna(x.mean()))
df.drop(['Z_CostContact', 'Z_Revenue' , "ID"], axis=1, inplace=True)
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
df['Age'] = df.Year_Birth.apply(lambda x: 2023 - int(x))
df.drop(columns=["Year_Birth"],axis=1 , inplace=True)
df['Education'].replace(['PhD', 'Master','2n Cycle'],'PostGraduate', inplace=True)  
df['Education'].replace(['Graduation'], 'Graduate', inplace=True)
df['Education'].replace(['Basic'],'UnderGraduate', inplace=True)
df["Children"]=df["Kidhome"]+df["Teenhome"]
df.drop(columns=["Kidhome" , "Teenhome"],axis=1 , inplace=True)
categorical_data = df[['Education', 'Marital_Status']]
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
df=df.rename(columns={"NumDealsPurchases": "Deals","NumWebPurchases":"Web","NumCatalogPurchases":"Catalog","NumStorePurchases":"Store","NumWebVisitsMonth":"WebVisits"})
df=df.rename(columns={"AcceptedCmp1": "A1","AcceptedCmp2":"A2","AcceptedCmp3":"A3","AcceptedCmp4":"A4","AcceptedCmp5":"A5"})
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = df['Dt_Customer'].max()
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Time_Enrolled_Days'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df=df.drop(columns=['Dt_Customer','Date_Collected'],axis=1)
df['ActiveDays'] = df['Time_Enrolled_Days'] - df['Recency']
df["Family_Size"] = df["Marital_Status"].replace({"Single": 1, "Couple":2}) + df["Children"]
df.drop(df[(df['Income'] > (df['Income'].mean() * 2)) | (df['Age'] > (df['Age'].mean() * 2))].index, inplace=True)
df['AM.Total'] = df['Wines']+df['Fruits']+df['Meat']+df['Fish']+df['Sweets']+df['Gold']
df['AllPurchase'] = df['Web']+df['Catalog']+df['Store']
df['Average.P'] = round((df['AM.Total'] / df['AllPurchase']), 1)
df['ShareDealsPurchase'] = round((df['Deals'] / df['AllPurchase']) * 100, 1)
df['TotalAcceptedCmp'] = df['A1']+df['A2']+df['A3']+df['A4']+df['A5']
df=df.drop(columns=['A1','A2','A3','A4','A5'],axis=1)
df = df[df["AllPurchase"] != 0]
df.drop(df[df['Average.P'] > (df['Average.P'].mean() * 4)].index, inplace=True)
df.drop(columns=["Average.P"],axis=1 , inplace=True)
X= df.drop('Response', axis=1).values
y =df['Response'].values
ct = ColumnTransformer([
    ('catagoric1', OneHotEncoder(), [0]),
    ('catagoric2', OneHotEncoder(), [1]),
    ('numeric', StandardScaler(), list(range(2, len(X.T))))])
scaled_ds = ct.fit_transform(X)
print('Preprocessed Data:')
print(scaled_ds[0])
print(scaled_ds.shape)


# In[ ]:


def f5():
    X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
    sm = SMOTE()
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train = pd.DataFrame(X_bal)
    df_train['Response'] = y_bal
    df_train['Random_Mask'] = True
    df_train['New_Response']=df_train['Response']
    df_train_main=df_train.copy()
    df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
    df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
    df_train_labeled=df_train[df_train['New_Response']!=-1]        
    X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
    y_train_labeled = df_train_labeled['New_Response']
    param_dist = param_dist = {
        "alpha": [0.0001, 0.001, 0.01, 0.00001],
        "learning_rate": ["constant", "adaptive"],
        "hidden_layer_sizes": [(100,),(50,), (50, 50), (200,), (100, 100), (200, 100)]}

    model = MLPClassifier()
    random_search = RandomizedSearchCV(estimator=model,
    param_distributions=param_dist)
    random_search.fit(X_train_labeled, y_train_labeled)
    modelpe = MLPClassifier(**random_search.best_params_)
    modelpe.fit(X_train_labeled, y_train_labeled)
    y_pred = modelpe.predict(X_test)
    auc_scores = roc_auc_score(y_test, y_pred)
    F1s = precision_score(y_test, y_pred)
    F1ss = f1_score(y_test, y_pred)
          
    x_values = np.arange(0.03, 0.50, 0.03)
    x_values = np.append(x_values, 0.49)
    scores = np.empty((x_values.shape[0],))
    scoresf =np.empty((x_values.shape[0],))
    scoresac =np.empty((x_values.shape[0],))
    for i, j in enumerate(x_values):
        df_train_main.loc[df_train_main.sample(frac=j).index, 'Random_Mask'] = False
        df_train_main['New_Response']= df_train_main.apply(lambda x:x['Response']if x['Random_Mask'] == False else -1,axis=1)
        X_train = df_train_main.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
        df_train_labeled=df_train[df_train['New_Response']!=-1]
        param_dist = {
        "alpha": [0.0001, 0.001, 0.01, 0.00001],
        "learning_rate": ["constant", "adaptive"],
        "hidden_layer_sizes": [(100,),(50,), (50, 50), (200,), (100, 100), (200, 100)]}

        model = MLPClassifier()
        random_search = RandomizedSearchCV(estimator=model,
        param_distributions=param_dist)
        random_search.fit(X_train_labeled, y_train_labeled)        
        modelpe = MLPClassifier(**random_search.best_params_ )       
        y_train = df_train_main['New_Response']
        
        x_t = np.arange(0.4, 1.05, 0.05)
        x_t = np.append(x_values, 0.99999)
        scorest = np.empty((x_values.shape[0],))
        for k, threshold in enumerate(x_t):
            self_training_clf = SelfTrainingClassifier(modelpe, threshold=threshold)
            self_training_clf.fit(X_train, y_train)
            y_predt = self_training_clf.predict(X_test)
            scorest[i] = roc_auc_score(y_test, y_pred)
        max_scoret = np.max(scores)
        max_threshold = x_values[np.argmax(scores)]  
        self_training_clf = SelfTrainingClassifier(modelpe, threshold=max_threshold)
        self_training_clf.fit(X_train, y_train)
        y_pred = self_training_clf.predict(X_test)
        scores[i] = roc_auc_score(y_test, y_pred)
        scoresf[i] = precision_score(y_test, y_pred)
        scoresac[i] = f1_score(y_test, y_pred)
    return (x_values , scores , scoresf , scoresac , auc_scores , F1s ,  F1ss , X_test, y_test)


# In[ ]:


results5 = []
for i in range(10):
    result = f5()
    results5.append(result)
print("Results:", results5)


# In[ ]:


mn_scoresA = np.mean([results5[i][1] for i in range(10)], axis=0)
mn_scoresFB = np.mean([results5[i][2] for i in range(10)], axis=0)
mn_scoresF1 = np.mean([results5[i][3] for i in range(10)], axis=0)
std_score = np.std([results5[i][3] for i in range(10)], axis=0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results5[0][0], results5[0][3], label='Scores1')
ax.plot(results5[0][0], results5[1][3], label='Scores2')
ax.plot(results5[0][0], results5[2][3], label='Scores3')
ax.plot(results5[0][0], results5[3][3], label='Scores4')
ax.plot(results5[0][0], results5[4][3], label='Scores5')
ax.plot(results5[0][0], results5[5][3], label='Scores6')
ax.plot(results5[0][0], results5[6][3], label='Scores7')
ax.plot(results5[0][0], results5[7][3], label='Scores8')
ax.plot(results5[0][0], results5[8][3], label='Scores9')
ax.plot(results5[0][0], results5[9][3], label='Scores10')
ax.fill_between(results5[0][0], mn_scoresF1-std_score, mn_scoresF1+std_score, alpha=0.2)
ax.legend()
ax.set_title('Shuffle Scores')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('Accuracy score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCf = np.mean([i[3] for i in results5], axis=0)
for i in range(len(results5[0][0])):
    mean_SVCsupervised = np.mean([i[6] for i in results5], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[3] for i in results5], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results5[0][0], mean_SVCf, label='self-training F1' , color='red')
ax.fill_between(results5[0][0], mean_SVCf-std_score/2, mean_SVCf+std_score/2, alpha=0.1 , color='red')
ax.plot(results5[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('ML Perceptron')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('F1 score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVCp = np.mean([i[2] for i in results5], axis=0)
for i in range(len(results5[0][0])):
    mean_SVCsupervised = np.mean([i[5] for i in results5], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[2] for i in results5], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results5[0][0], mean_SVCp, label='self-training F1' , color='blue')
ax.fill_between(results5[0][0], mean_SVCp-std_score/2, mean_SVCp+std_score/2, alpha=0.1 , color='blue')
ax.plot(results5[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('ML Perceptron')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('precision score')
plt.show()


# In[ ]:


mean_SVCf = np.mean([i[3] for i in results5], axis=0)
mean_SVCp = np.mean([i[2] for i in results5], axis=0)
std_SVCf = np.std([i[3] for i in results5], axis=0)
std_SVCp = np.std([i[2] for i in results5], axis=0)

w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresF1)]
z=[]
mean_SVCr5=(mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))
for i in range(len(results6[0][0])):
    mean_SVCsupervised = np.mean([(i[6]*i[5]/((2*i[5])-i[6])) for i in results5], axis=0)
    z.append(mean_SVCsupervised)
std_score=(std_SVCf*std_SVCp/((2*std_SVCp)-std_SVCf))

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results5[0][0], mean_SVCr5, label='self-training recall' , color='red')
ax.fill_between(results5[0][0], mean_SVCr5-std_score/2, mean_SVCr5+std_score/2, alpha=0.1 , color='red')
ax.plot(results5[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('ML Perceptron')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('recall score')
plt.show()


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]))]
z=[]
mean_SVC = np.mean([i[1] for i in results5], axis=0)
for i in range(len(results5[0][0])):
    mean_SVCsupervised = np.mean([i[4] for i in results5], axis=0)
    z.append(mean_SVCsupervised)
std_score = np.std([i[1] for i in results5], axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(results5[0][0], mean_SVC, label='self-training F1' , color='brown')
ax.fill_between(results5[0][0], mean_SVC-std_score/2, mean_SVC+std_score/2, alpha=0.1 , color='brown')
ax.plot(results5[0][0], z, label='supervised F1 ', color='green', linestyle='--')

ax.legend()
ax.set_title('ML Perceptron')
ax.set_xlabel('Labeled data percentage')
ax.set_ylabel('ROC-AUC score')
plt.show()


# In[ ]:


a=[]
b=[]
c=[]
mean_listauc = [] 
mean_listf1 = [] 
mean_listacc = [] 
for i in range(10):
    mean_val1 = (results5[i][1][7])
    mean_val2 = (results5[i][2][7])
    mean_val3 = (results5[i][3][7])
    aa=resultsss[i][21]
    bb=resultsss[i][22]
    cc=resultsss[i][23]
    a.append(aa)
    b.append(bb)
    c.append(cc)   
    mean_listauc.append(mean_val1)
    mean_listf1.append(mean_val3)
    mean_listacc.append(mean_val2)
    
mean_supervisedauc = np.mean(a)
mean_supervisedpre = np.mean(b)
mean_supervisedf1 = np.mean(c)
    
mean_self_trainauc=np.mean(mean_listauc)
mean_self_trainfpre=np.mean(mean_listacc)
mean_self_trainf1=np.mean(mean_listf1)


print("Self-training AUC score with Perceptron  classifier is:", "{:.3f}%".format(mean_self_trainauc * 100))
print("Supervised Perceptron  classifier model AUC score is:", "{:.3f}%".format(mean_supervisedauc * 100))
if mean_self_trainauc > mean_supervisedauc:
    print("Self-training AUC score with Perceptron  classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training AUC score with Perceptron  classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training precision with Perceptron  classifier is:", "{:.3f}%".format(mean_self_trainfpre * 100))
print("Supervised Perceptron  model precision is:", "{:.3f}%".format(mean_supervisedpre * 100))
if mean_self_trainfpre > mean_supervisedpre:
    print("Self-training precision with Perceptron  classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training precision with Perceptron  classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training F1 with logistic Perceptron  is:", "{:.3f}%".format(mean_self_trainf1 * 100))
print("Supervised Perceptron  model accuracy is:", "{:.3f}%".format(mean_supervisedf1 * 100))
if mean_self_trainf1 > mean_supervisedf1:
    print("Self-training F1 with Perceptron  classifier is greater than supervised model for 24% of labeled data")
else:
    print("Self-training F1 with Perceptron  classifier is not greater than supervised model for 24% of labeled data")
print("-------------------------------------------------------------------------------------")   
print("Self-training recall with Perceptron  is:", "{:.3f}%".format(((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1)))  * 100))
print("Supervised Perceptron  model recall is:", "{:.3f}%".format(((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))* 100))
if (((mean_self_trainf1*mean_self_trainfpre/((2*mean_self_trainfpre)-mean_self_trainf1))))  > (((mean_supervisedf1*mean_supervisedpre/((2*mean_supervisedpre)-mean_supervisedf1)))):
    print("Self-training recall with Perceptron   is greater than supervised model for 24% of labeled data")
else:
    print("Self-training recall with Perceptron   is not greater than supervised model for 24% of labeled data")


# In[ ]:


w = [(np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 , 0.33,
       0.36, 0.39, 0.42, 0.45, 0.48, 0.49]), mn_scoresFB, mn_scoresF1,mn_scoresA)]
mean_SVC = np.mean([i[1] for i in results5], axis=0)
    
max_index = np.argmax(w[0][1]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][1][max_index]
print("maximum  precision of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum precision:", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][2]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][2][max_index]
print("maximum F1 of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum F1 :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax(w[0][3]) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = w[0][3][max_index]
print("maximum AUC  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum AUC :", "{:.1f}%".format(max_value_first_row*100))

max_index = np.argmax((mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))) 
max_value_first_row = w[0][0][max_index] 
max_value_second_row = (mean_SVCf*mean_SVCp/((2*mean_SVCp)-mean_SVCf))[max_index]
print("maximum recall  of self-training model:", "{:.3f}%".format(max_value_second_row*100))
print("amount of labeled data for maximum recall :", "{:.1f}%".format(max_value_first_row*100))


# In[ ]:


y1 = np.mean([results1[i][2] for i in range(10)], axis=0)
y2 = np.mean([results2[i][2] for i in range(10)], axis=0)
y3 = np.mean([results3[i][2] for i in range(10)], axis=0)
y4 = np.mean([results4[i][2] for i in range(10)], axis=0)
y5 = np.mean([results5[i][2] for i in range(10)], axis=0)
y6 = np.mean([results6[i][2] for i in range(10)], axis=0)
y7 = np.mean([results77[i][2] for i in range(10)], axis=0)
y8 = np.mean([results8[i][2] for i in range(10)], axis=0)
x  = results1[0][0]
df = pd.DataFrame({'x': x, 'csv': y1, 'logistic': y2, 'randomforest': y3 , 'naive': y4, 
                   'perceptron': y5, 'knn': y6, 'decission tree': y7 , 'gradiant': y8})
df_melted = pd.melt(df, id_vars=['x'], value_vars=['csv', 'logistic', 'randomforest', 'naive' , 'perceptron',
                                                  'knn' ,'gradiant' , "decission tree" ], var_name='Trend')
sns.set_style('ticks')
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(10, 6))
custom_colors = ['#C0C0C0', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFFF00','#FF0000' , '#800080']

sns.lineplot(x='x', y='value', hue='Trend', data=df_melted, palette=custom_colors, ax=ax)
ax.set_title('Comparison of classifiers', fontsize=20, fontweight='bold')
ax.set_xlabel('labeled data', fontsize=16)
ax.set_ylabel('Precision Scores', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(title=None, loc='lower right', fontsize=12)

ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
sns.despine()
plt.show()


# In[ ]:


y1 = np.mean([results1[i][1] for i in range(10)], axis=0)
y2 = np.mean([results2[i][1] for i in range(10)], axis=0)
y3 = np.mean([results3[i][1] for i in range(10)], axis=0)
y4 = np.mean([results4[i][1] for i in range(10)], axis=0)
y5 = np.mean([results5[i][1] for i in range(10)], axis=0)
y6 = np.mean([results6[i][1] for i in range(10)], axis=0)
y7 = np.mean([results77[i][1] for i in range(10)], axis=0)
y8 = np.mean([results8[i][1] for i in range(10)], axis=0)
x  = results1[0][0]

df = pd.DataFrame({'x': x, 'csv': y1, 'logistic': y2, 'randomforest': y3 , 'naive': y4, 
                   'perceptron': y5, 'knn': y6, 'decission tree': y7 , 'gradiant': y8})
df_melted = pd.melt(df, id_vars=['x'], value_vars=['csv', 'logistic', 'randomforest', 'naive' , 'perceptron',
                                                  'knn' ,'gradiant' , "decission tree" ], var_name='Trend')
sns.set_style('ticks')
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(10, 6))
custom_colors = ['#C0C0C0', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFFF00','#FF0000' , '#800080']

sns.lineplot(x='x', y='value', hue='Trend', data=df_melted, palette=custom_colors, ax=ax)
ax.set_title('Comparison of classifiers', fontsize=20, fontweight='bold')
ax.set_xlabel('labeled data', fontsize=16)
ax.set_ylabel('AUC Scores', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(title=None, loc='lower right', fontsize=12)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
sns.despine()
plt.show()


# In[ ]:


y1 = np.mean([results1[i][3] for i in range(10)], axis=0)
y2 = np.mean([results2[i][3] for i in range(10)], axis=0)
y3 = np.mean([results3[i][3] for i in range(10)], axis=0)
y4 = np.mean([results4[i][3] for i in range(10)], axis=0)
y5 = np.mean([results5[i][3] for i in range(10)], axis=0)
y6 = np.mean([results6[i][3] for i in range(10)], axis=0)
y7 = np.mean([results77[i][3] for i in range(10)], axis=0)
y8 = np.mean([results8[i][3] for i in range(10)], axis=0)
x  = results1[0][0]
df = pd.DataFrame({'x': x, 'csv': y1, 'logistic': y2, 'randomforest': y3 , 'naive': y4, 
                   'perceptron': y5, 'knn': y6, 'decission tree': y7 , 'gradiant': y8})
df_melted = pd.melt(df, id_vars=['x'], value_vars=['csv', 'logistic', 'randomforest', 'naive' , 'perceptron',
                                                  'knn' ,'gradiant' , "decission tree" ], var_name='Trend')
sns.set_style('ticks')
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(10, 6))
custom_colors = ['#C0C0C0', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFFF00','#FF0000' , '#800080']

sns.lineplot(x='x', y='value', hue='Trend', data=df_melted, palette=custom_colors, ax=ax)
ax.set_title('Comparison of classifiers', fontsize=20, fontweight='bold')
ax.set_xlabel('labeled data', fontsize=16)
ax.set_ylabel('F1 Scores', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(title=None, loc='lower right', fontsize=12)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
sns.despine()
plt.show()


# In[ ]:


y1 = mean_SVCr1
y2 = mean_SVCr2
y3 = mean_SVCr3
y4 = mean_SVCr4
y5 = mean_SVCr5
y6 = mean_SVCr6
y7 = mean_SVCr7
y8 = mean_SVCr8
x  = results1[0][0]
df = pd.DataFrame({'x': x, 'SVM': y1, 'logistic': y2, 'randomforest': y3 , 'naive': y4, 
                   'perceptron': y5, 'KNN': y6, 'decission tree': y7 , 'gradiant': y8})
df_melted = pd.melt(df, id_vars=['x'], value_vars=['SVM', 'logistic', 'randomforest', 'naive' , 'perceptron',
                                                  'KNN' ,'gradiant' , "decission tree" ], var_name='Trend')
sns.set_style('ticks')
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(10, 6))
custom_colors = [ '#0000FF', '#00FF00','#C0C0C0', '#FF00FF', '#00FFFF', '#FFFF00','#FF0000' , '#800080']

sns.lineplot(x='x', y='value', hue='Trend', data=df_melted, palette=custom_colors, ax=ax)
ax.set_title('Comparison of classifiers', fontsize=20, fontweight='bold')
ax.set_xlabel('labeled data', fontsize=16)
ax.set_ylabel('recall Scores', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.legend(title=None, loc='lower left', fontsize=8)
sns.despine()
plt.show()


# In[ ]:


y1 = np.mean([results1[i][1] for i in range(10)], axis=0)
y2 = np.mean([results2[i][1] for i in range(10)], axis=0)
y3 = np.mean([results3[i][1] for i in range(10)], axis=0)

y5 = np.mean([results5[i][1] for i in range(10)], axis=0)
y6 = np.mean([results6[i][1] for i in range(10)], axis=0)
y7 = np.mean([results77[i][1] for i in range(10)], axis=0)
y8 = np.mean([results8[i][1] for i in range(10)], axis=0)
x  = results1[0][0]

df = pd.DataFrame({'x': x, 'csv': y1, 'logistic': y2, 'randomforest': y3 ,  
                   'perceptron': y5, 'knn': y6, 'decission tree': y7  , 'Gradient Boosting': y8 })
df_melted = pd.melt(df, id_vars=['x'], value_vars=['csv', 'logistic', 'randomforest' , 'perceptron',
                                                  'knn'  , "decission tree" , 'Gradient Boosting' ], var_name='Trend')
sns.set_style('ticks')
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette('colorblind', n_colors=len(df_melted['Trend'].unique()))
sns.lineplot(x='x', y='value', hue='Trend', data=df_melted, palette=palette, ax=ax)
ax.set_title('Comparison of classifiers', fontsize=20, fontweight='bold')
ax.set_xlabel('labeled data', fontsize=16)
ax.set_ylabel('AUC Scores', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(title=None, loc='lower right', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.show()


# In[ ]:


y1 = np.mean([results1[i][2] for i in range(10)], axis=0)
y2 = np.mean([results2[i][2] for i in range(10)], axis=0)
y3 = np.mean([results3[i][2] for i in range(10)], axis=0)

y5 = np.mean([results5[i][2] for i in range(10)], axis=0)
y6 = np.mean([results6[i][2] for i in range(10)], axis=0)
y7 = np.mean([results77[i][2] for i in range(10)], axis=0)
y8 = np.mean([results8[i][2] for i in range(10)], axis=0)
x  = results1[0][0]

df = pd.DataFrame({'x': x, 'csv': y1, 'logistic': y2, 'randomforest': y3 , 
                   'perceptron': y5, 'knn': y6, 'decission tree': y7 , 'Gradient Boosting': y8 })
df_melted = pd.melt(df, id_vars=['x'], value_vars=['csv', 'logistic', 'randomforest', 'perceptron',
                                                  'knn'  , "decission tree" , 'Gradient Boosting'], var_name='Trend')
sns.set_style('ticks')
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette('colorblind', n_colors=len(df_melted['Trend'].unique()))
sns.lineplot(x='x', y='value', hue='Trend', data=df_melted, palette=palette, ax=ax)
ax.set_title('Comparison of classifiers', fontsize=20, fontweight='bold')
ax.set_xlabel('labeled data', fontsize=16)
ax.set_ylabel('Precision', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(title=None, loc='lower right', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.show()


# In[ ]:


y1 = np.mean([results1[i][3] for i in range(10)], axis=0)
y2 = np.mean([results2[i][3] for i in range(10)], axis=0)
y3 = np.mean([results3[i][3] for i in range(10)], axis=0)

y5 = np.mean([results5[i][3] for i in range(10)], axis=0)
y6 = np.mean([results6[i][3] for i in range(10)], axis=0)
y7 = np.mean([results77[i][3] for i in range(10)], axis=0)
y8 = np.mean([results8[i][3] for i in range(10)], axis=0)
x  = results1[0][0]

df = pd.DataFrame({'x': x, 'csv': y1, 'logistic': y2, 'randomforest': y3 , 
                   'perceptron': y5, 'knn': y6, 'decission tree': y7 , 'Gradient Boosting' : y8 })
df_melted = pd.melt(df, id_vars=['x'], value_vars=['csv', 'logistic', 'randomforest', 'perceptron',
                                                  'knn' , "decission tree" , 'Gradient Boosting' ], var_name='Trend')
sns.set_style('ticks')
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(10, 6))
palette = sns.color_palette('colorblind', n_colors=len(df_melted['Trend'].unique()))
sns.lineplot(x='x', y='value', hue='Trend', data=df_melted, palette=palette, ax=ax)
ax.set_title('Comparison of classifiers', fontsize=20, fontweight='bold')
ax.set_xlabel('labeled data', fontsize=16)
ax.set_ylabel('F1 Scores', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(title=None, loc='lower right', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
sns.set(font_scale=1.5)
X_train, X_test, y_train, y_test = train_test_split(scaled_ds, y, test_size=0.25)
sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X_train, y_train)
df_train = pd.DataFrame(X_bal)
df_train['Response'] = y_bal
df_train['Random_Mask'] = True
df_train['New_Response']=df_train['Response']
df_train_main=df_train.copy()
df_train.loc[df_train.sample(frac=1).index, 'Random_Mask'] = False
df_train['New_Response']=df_train.apply(lambda x: x['Response'] if x['Random_Mask']==False else -1, axis=1)
df_train_labeled=df_train[df_train['New_Response']!=-1]        
X_train_labeled = df_train_labeled.drop(['Response', 'Random_Mask', 'New_Response'], axis=1)
y_train_labeled = df_train_labeled['New_Response']

param_dist_svc = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['poly', 'rbf', 'linear', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'coef0': [-1, 0, 1]
}
model_svc = SVC()
random_search_svc = RandomizedSearchCV(estimator=model_svc, param_distributions=param_dist_svc, scoring='roc_auc')
random_search_svc.fit(X_train_labeled, y_train_labeled)
model_svc_best = SVC(**random_search_svc.best_params_, probability=True)
model_svc_best.fit(X_train_labeled, y_train_labeled)

fig, axes = plt.subplots(4, 2 , figsize=(12, 18), sharex=True , sharey=True )

plot_confusion_matrix(model_svc_best, X_test, y_test, cmap='Blues', normalize='true',
                      display_labels=['Response', 'No Response'], ax=axes[0,0])
axes[0, 0].grid(False)
axes[0,0].set_title('SVC')

param_dist_logreg = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
}
model_logreg = LogisticRegression()
random_search_logreg = RandomizedSearchCV(estimator=model_logreg, param_distributions=param_dist_logreg, scoring='roc_auc')
random_search_logreg.fit(X_train_labeled, y_train_labeled)
model_logreg_best = LogisticRegression(**random_search_logreg.best_params_)
model_logreg_best.fit(X_train_labeled, y_train_labeled)

plot_confusion_matrix(model_logreg_best, X_test, y_test, cmap='YlGnBu', normalize='true',
                      display_labels=['Response', 'No Response'], ax=axes[0,1])
axes[0, 1].grid(False)
axes[0,1].set_title('Logistic Regression')

param_dist =  {
    'n_estimators': [50, 100, 200, 300],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']}

model = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modelrfs = RandomForestClassifier(**random_search.best_params_)
modelrfs.fit(X_train_labeled, y_train_labeled)
plot_confusion_matrix(modelrfs, X_test, y_test, cmap='YlOrBr', normalize='true',
                      display_labels=['Response', 'No Response'], ax=axes[1,0])
axes[1, 0].grid(False)
axes[1,0].set_title('RandomForestClassifier')
    
param_dist =  {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-10, 1e-11, 1e-12, 1e-4]}

model = GaussianNB()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modelnb = GaussianNB(**random_search.best_params_)
modelnb.fit(X_train_labeled, y_train_labeled)
    
plot_confusion_matrix(modelnb, X_test, y_test, cmap='Greens', normalize='true',
                      display_labels=['Response', 'No Response'], ax=axes[1,1])
axes[1, 1].grid(False)
axes[1,1].set_title('Naive bayes')
    
param_dist = {'n_neighbors': [5, 7, 9, 10], 'leaf_size': [20,30,50],
             'metric': ['euclidean', 'manhattan' , 'minkowski']}

model = KNeighborsClassifier()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modelkn = KNeighborsClassifier(**random_search.best_params_)
modelkn.fit(X_train_labeled, y_train_labeled)
    
plot_confusion_matrix(modelkn, X_test, y_test, cmap='Reds', normalize='true',
                      display_labels=['Response', 'No Response'], ax=axes[2,0])
axes[2, 0].grid(False)
axes[2,0].set_title('KNN')

param_dist = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [1 ,2, 3, 4, 5, 6, 7, 8],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8]}

model = DecisionTreeClassifier()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modelDT = DecisionTreeClassifier(**random_search.best_params_)
modelDT.fit(X_train_labeled, y_train_labeled)

plot_confusion_matrix(modelDT, X_test, y_test, cmap='Oranges', normalize='true',
                      display_labels=['Response', 'No Response'], ax=axes[2,1])
axes[2, 1].grid(False)
axes[2,1].set_title('DecisionTreeClassifier')

param_dist = {
    'n_estimators': sp_randint(50,100, 200),
    'learning_rate': [0.1, 0.01, 0.001 , 1],
    'max_depth': sp_randint(2, 10),
    'subsample': [0.5, 2, 1.0]}
    
model = GradientBoostingClassifier()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist,scoring='recall',cv=3)
random_search.fit(X_train_labeled, y_train_labeled)
df_train_labeled=df_train[df_train['New_Response']!=-1]
modelgb = GradientBoostingClassifier(**random_search.best_params_)
modelgb.fit(X_train_labeled, y_train_labeled)

plot_confusion_matrix(modelgb, X_test, y_test, cmap='gray', normalize='true',
                      display_labels=['Response', 'No Response'], ax=axes[3,0])
axes[3, 0].grid(False)
axes[3,0].set_title('GradientBoostingClassifier')
    
param_dist = {
        "alpha": [0.0001, 0.001, 0.01, 0.00001],
        "learning_rate": ["constant", "adaptive"],
        "hidden_layer_sizes": [(100,),(50,), (50, 50), (200,), (100, 100), (200, 100)]}

model = MLPClassifier()
random_search = RandomizedSearchCV(estimator=model,
param_distributions=param_dist , scoring='roc_auc')
random_search.fit(X_train_labeled, y_train_labeled)
modelpe = MLPClassifier(**random_search.best_params_)
modelpe.fit(X_train_labeled, y_train_labeled)

plot_confusion_matrix(modelpe, X_test, y_test, cmap='Purples', normalize='true',
                      display_labels=['Response', 'No Response'], ax=axes[3,1])
axes[3, 1].grid(False)
axes[3,1].set_title('Perceptron')
    
plt.show()


# In[ ]:




