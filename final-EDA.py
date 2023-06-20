#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from plotly import tools
from plotly.subplots import make_subplots
from plotly.offline import iplot


# In[ ]:


main_df=pd.read_csv("C:/Users/gebruiker/Desktop/New folder/marketing_campaign.csv", sep='\t')


# In[ ]:


main_df.head()


# In[ ]:


df = main_df.copy()


# In[ ]:


print("Shape of the dataFrame is :",df.shape)


# In[ ]:


print("Size of the dataFrame is :",df.size)


# In[ ]:


print("Dimensions of the dataFrame is :",df.ndim)


# In[ ]:


print("Column_name in the dataFrame is : \n ",df.columns)


# In[ ]:


print("Print a summary of a dataframe is :",df.info())


# In[ ]:


df.isna().sum()


# In[ ]:


df["Income"] = df.groupby("Education")['Income'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


df.isna().sum()


# In[ ]:


df.nunique().sort_values(0)


# In[ ]:



#Response  and  AcceptedCmp are all Binary Variables.
#Marital_Status & Education can be seen as Categorical Variable.
#The rest variables can be seen as Continuous Ordinal Variables.


# In[ ]:


#Z_Revenue & Z_CostContact have Constant value, which don't provide any information so we should drop them.
df.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True) 


# In[ ]:


df.drop(['ID'], axis=1, inplace=True)


# In[ ]:


df['Marital_Status'].value_counts()


# In[ ]:


df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')


# In[ ]:


df['Marital_Status'].value_counts()  


# In[ ]:


df['Age'] = 2023 - df["Year_Birth"]
df.drop(columns=["Year_Birth"],axis=1 , inplace=True)


# In[ ]:


main_df.head()


# In[ ]:


df['Education'].value_counts()


# In[ ]:


df['Education'].replace(['PhD', 'Master','2n Cycle'],'PostGraduate', inplace=True)  
df['Education'].replace(['Graduation'], 'Graduate', inplace=True)
df['Education'].replace(['Basic'],'UnderGraduate', inplace=True)


# In[ ]:


df['Education'].value_counts()


# In[ ]:


df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['Date_Collected'] = df['Dt_Customer'].max()
df['Date_Collected'] = pd.to_datetime(df.Date_Collected)
df['Time_Enrolled_Days'] = (df['Date_Collected'] - df['Dt_Customer']).dt.days
df=df.drop(columns=['Dt_Customer','Date_Collected'],axis=1)


# In[ ]:


df['ActiveDays'] = df['Time_Enrolled_Days'] - df['Recency']


# In[ ]:


df["Children"]=df["Kidhome"]+df["Teenhome"]
df.drop(columns=["Kidhome" , "Teenhome"],axis=1 , inplace=True)


# In[ ]:


df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
df=df.rename(columns={"NumDealsPurchases": "Deals","NumWebPurchases":"Web","NumCatalogPurchases":"Catalog","NumStorePurchases":"Store","NumWebVisitsMonth":"WebVisits"})
df=df.rename(columns={"AcceptedCmp1": "A1","AcceptedCmp2":"A2","AcceptedCmp3":"A3","AcceptedCmp4":"A4","AcceptedCmp5":"A5"})


# In[ ]:


df["Family_Size"] = df["Marital_Status"].replace({"Single": 1, "Couple":2}) + df["Children"]


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df['AM.Total'] = df['Wines']+df['Fruits']+df['Meat']+df['Fish']+df['Sweets']+df['Gold']
df['AllPurchase'] = df['Web']+df['Catalog']+df['Store']
df['Average.P'] = round((df['AM.Total'] / df['AllPurchase']), 1)
df['ShareDealsPurchase'] = round((df['Deals'] / df['AllPurchase']) * 100, 1)
df['TotalAcceptedCmp'] = df['A1']+df['A2']+df['A3']+df['A4']+df['A5']+df['Response']


# In[ ]:


df = df[df["AllPurchase"] != 0]


# In[ ]:


num_coln = ["Income", "Age", "Average.P"]
bins = 10

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))  # Update figsize if needed
plt.subplots_adjust(hspace=0.5)

for i, col in enumerate(num_coln):
    ax = axs[i]  # Use a single index instead of row and column indices
    sns.boxplot(data=df, x=col, ax=ax)
    ax.set_xlabel(col)

plt.show()


# In[ ]:


df.drop(df[(df['Income'] > (df['Income'].mean() * 2)) | (df['Age'] > (df['Age'].mean() * 2))].index, inplace=True) #outlier


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.nunique().sort_values()


# In[ ]:


num_coln = ["Income", "Recency", "Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Deals", "Web", "Catalog", "Store", "WebVisits", "Age", "Time_Enrolled_Days", "Children", "Family_Size" , "Average.P" ]
bins = 10

fig, axs = plt.subplots(nrows=9, ncols=2, figsize=(20, 30))
plt.subplots_adjust(hspace=0.5)

for i, col in enumerate(num_coln):
    row, col_idx = divmod(i, 2)
    ax = axs[row, col_idx]
    sns.boxplot(data=df, x=col, ax=ax)
    ax.set_xlabel(col)

plt.show()


# In[ ]:


df.drop(df[df['Average.P'] > (df['Average.P'].mean() * 2)].index, inplace=True)


# In[ ]:


sns.boxplot(x=df['Income'])
plt.xlabel('Income')
plt.show()


# In[ ]:


sns.boxplot(x=df['Average.P'])
plt.xlabel('Average.P')
plt.show()


# In[ ]:


df.nunique().sort_values()


# In[ ]:


df.shape


# In[ ]:


cust_count=df.groupby("Marital_Status").count()['Income']
label=df.groupby('Marital_Status').count()['Income'].index
fig, ax = plt.subplots(1, 2, figsize = (10, 12))
ax[0].pie(cust_count, labels=label, shadow=True, autopct='%1.2f%%',radius=2,explode=[0.3,0.2],colors=['#ff6961' , '#66b3ff'])
ax[0].set_title('Maritial Status', y=-0.6)

cust_count = df.groupby("Education").count()['Income']
label = df.groupby('Education').count()['Income'].index
ax[1].pie(cust_count, labels=label, shadow=True, autopct='%1.2f%%',radius=2,explode=[0.3,0.2,0.1],colors=['#ff9999' , '#00b894' , '#1e90ff'])
ax[1].set_title('Education Level', y=-0.6)
plt.subplots_adjust(wspace = 1.5, hspace =0)
plt.show()


# In[ ]:


fig = px.box(df, x="Web", y="Store")
fig.show()


# In[ ]:


fig = px.scatter(df, x="Web", y="WebVisits")
fig.show()


# In[ ]:


fig = px.histogram(df, x="Age", nbins=30, color='Age', title="Distribution of Age")
fig.show()


# In[ ]:


import plotly.express as px

# Select the features for the scatter plot
features = ['Education', 'Marital_Status', 'Income', 'Recency', 'Wines', 'Fruits',
            'Meat', 'Fish', 'Sweets', 'Gold', 'Deals', 'Web', 'Catalog', 'Store',
            'WebVisits', 'Complain', 'Response', 'Age', 'Children',
            'Time_Enrolled_Days', 'ActiveDays', 'Family_Size', 'AM.Total',
            'AllPurchase', 'ShareDealsPurchase', 'TotalAcceptedCmp']

# Create the scatter plot
fig = px.scatter(df, x='ActiveDays', y='Income',size =  'AllPurchase', color='Response', hover_data=features)
plt.figure(figsize=(25, 23)) 
# Customize the layout and appearance
fig.update_layout(
    title="Scatter Plot of Response vs. Income",
    xaxis_title="ActiveDays",
    yaxis_title="Income"
)

# Show the plot
fig.show()


# In[ ]:


plt.figure(figsize=(20, 6))
plt.title('Age distribution')
ax = sns.histplot(df['Age'].sort_values(), bins=56 ,color='green')
plt.xticks(np.linspace(df['Age'].min(), df['Age'].max(), 56, dtype=int, endpoint = True))
plt.show()


# In[ ]:


plt.figure(figsize=(16, 8))
sns.set_style('whitegrid')
plt.title('Income', fontsize=18)
plt.xlabel('Income', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
ax = sns.distplot(df['Income'].sort_values(), bins=40, color='BLUE')
plt.xticks(np.linspace(df['Income'].min(), df['Income'].max(), 20, dtype=int, endpoint=True), fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[ ]:


plt.figure(figsize=(22, 6))
mean, median = df["Income"].mean(), df["Income"].median()
graph = sns.histplot(df["Income"])
graph.axvline(mean, c='red',label='mean')
graph.axvline(median, c='green',label='median')
plt.legend()
print("mean of income" , df["Income"].mean())
print("median of income" , df["Income"].median())


# In[ ]:


sns.kdeplot(data=df, x="Income", shade=True, log_scale=True )
plt.xlim(2000, 130000)
plt.show()


# In[ ]:


plt.figure(figsize=(40,10))
plt.subplot(121)
sns.histplot(data=df, x="Income", hue="TotalAcceptedCmp", multiple="stack")
plt.show()


# In[ ]:


plt.figure(figsize=(10, 8))  # Adjust the figure size here
sns.set(font_scale=1.5) 
sns.kdeplot(
    data=df, x="Income", hue="TotalAcceptedCmp",
    fill=True, common_norm=False, palette="Paired",
    alpha=0.7, linewidth=0,)
plt.gca().axes.get_yaxis().set_visible(False)
plt.xlabel('Income')
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(221)
sns.kdeplot(
   data=df, x="Income", hue="TotalAcceptedCmp", log_scale= True,
   fill=True, common_norm=False, palette="Paired" ,
   alpha=.5, linewidth=0,
)
plt.gca().axes.get_yaxis().set_visible(False)
plt.xlabel('Income')

plt.subplot(222)
sns.kdeplot(
   data=df, x="Income", hue="Marital_Status", log_scale= True,
   fill=True, common_norm=False, palette="Set1",
   alpha=.5, linewidth=0,
)
plt.gca().axes.get_yaxis().set_visible(False) 

plt.subplot(223)
sns.kdeplot(
   data=df, x="Income", hue="Education", log_scale= True,
   fill=True, common_norm=False, palette="magma",
   alpha=.5, linewidth=0,
)
plt.gca().axes.get_yaxis().set_visible(False) 

plt.subplot(224)
sns.kdeplot(
   data=df, x="Income", hue="Children", log_scale= True,
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)
plt.gca().axes.get_yaxis().set_visible(False)


# In[ ]:


df.nunique().sort_values(0)


# In[ ]:


plt.figure(figsize=(20,20))
to_plot = ['Age','Recency','AM.Total',"AllPurchase",'Income',"Time_Enrolled_Days"]
sns.pairplot(df[to_plot], hue='Income', palette='husl')
plt.show()


# In[ ]:


sns.set(font_scale=1.5)  # Set the font scale for the plot

plt.figure(figsize=(20, 20))

to_plot = ['Response', 'TotalAcceptedCmp', 'ActiveDays', 'Average.P', 'AM.Total', 'Income']

sns.pairplot(df[to_plot], hue='Response', palette='husl')

plt.xlabel('X Label')
plt.ylabel('Y Label')

plt.show()


# In[ ]:


cor = df.corr()
plt.figure(figsize = (27,26))
sns.heatmap(cor, annot = True, cmap = 'viridis')
plt.show()


# In[ ]:


cor = df.corr()
response_corr = cor.loc[['Response'], :]
response_corr_sorted = response_corr.sort_values(by='Response', axis=1)
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(data=response_corr_sorted, orient='h', palette='Set2')
for i, v in enumerate(response_corr_sorted.values.flatten()):
    bar_plot.text(v + 0.01, i, str(round(v, 2)), color='black', va='center')
plt.show()


# In[ ]:



cor = df.corr()
response_corr = cor.loc[['Response'], :]

plt.figure(figsize=(20, 1))
sns.heatmap(response_corr, annot=True, cmap='viridis')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 10))
  
sns.barplot(x=df.groupby("Marital_Status").mean()["Income"].index, y="Income", data=df.groupby("Marital_Status").mean(), ci=None, ax=axes[0, 0])
sns.barplot(x=df.groupby("TotalAcceptedCmp").mean()["Income"].index, y="Income", data=df.groupby("TotalAcceptedCmp").mean(), ci=None, ax=axes[0, 1])
sns.barplot(x=df.groupby("Web").mean()["Income"].index, y="Income", data=df.groupby("Web").mean(), ci=None ,ax=axes[1, 0])
sns.barplot(x=df.groupby("Store").mean()["Income"].index, y="Income", data=df.groupby("Store").mean(), ci=None, ax=axes[1, 1])
sns.barplot(x=df.groupby("Catalog").mean()["Income"].index, y="Income", data=df.groupby("Catalog").mean(), ci=None, ax=axes[2, 0])
sns.barplot(x=df.groupby("Family_Size").mean()["Income"].index, y="Income", data=df.groupby("Family_Size").mean(), ci=None, ax=axes[2, 1])
sns.barplot(x=df.groupby("Deals").mean()["Income"].index, y="Income", data=df.groupby("Deals").mean(), ci=None, ax=axes[3, 0])
sns.barplot(x=df.groupby("Education").mean()["Income"].index, y="Income", data=df.groupby("Education").mean(), ci=None, ax=axes[3, 1])

plt.tight_layout()
plt.show()


# In[ ]:


df.dtypes


# In[ ]:


df.nunique().sort_values()


# In[ ]:


df.isna().sum()


# In[ ]:


df.info()


# In[ ]:


df.nunique().sort_values()


# In[ ]:


df=df.drop(columns=['A1','A2','A3','A4','A5'],axis=1)


# In[ ]:


df.info()


# In[ ]:


df.Response.value_counts()


# In[ ]:


resp = df.Response.sum()
total = df.shape[0]
percent = round((resp / total)*100, 2)

print(resp, 'customers responded to the marketing campaign out of a total of', total, 'cutomers.')
print('Percent Responded: ' +  str(percent) + '%')


# In[ ]:


from tabulate import tabulate
import pandas as pd
import numpy as np

response_corr_abs = np.abs(df.corr()['Response']).sort_values(ascending=False)[1:]
response_corr = df.corr()['Response'].sort_values(ascending=False)[1:]

# Create a DataFrame to store the correlation coefficients
corr_df = pd.DataFrame({'Feature': response_corr.index, 'Correlation': response_corr.values, 'Absolute Correlation': response_corr_abs.values})

# Convert the DataFrame to a table format
table = tabulate(corr_df, headers='keys', tablefmt='fancy_grid')

# Print the table
print("Correlation Coefficients for 'Response'")
print(table)


# In[ ]:


def display_kdeplot(df, col1, col2):
    plt.figure(figsize=(8,8))
    sns.kdeplot(data=df, x=col1, hue=col2, multiple="stack")
    plt.title(col1 + ' correlation with ' + col2)
    plt.legend(['Response', 'No Response'])
    plt.show()
    print(df[col1].describe())


# In[ ]:


for i in range(5):
    feature_name = response_corr_abs.index[i]
    display_kdeplot(df, feature_name, 'Response')
    print('Correlation %:', round(response_corr[feature_name] * 100, 2))
    print('------------------------------------------------------------------------')


# In[ ]:


s = (df.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)


# In[ ]:


categorical_data = df[['Education', 'Marital_Status']]
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(categorical_data)
encoded_data = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names(['Education', 'Marital_Status']))
df = pd.concat([df, encoded_data], axis=1)
df.drop(['Education', 'Marital_Status'], axis=1, inplace=True)


# In[ ]:


unique_values = df['Response'].value_counts()
print(unique_values)
print(unique_values/df.shape)


# In[ ]:


ds = df.copy()
ds.info()


# In[ ]:


#remove target:
ds.drop('Income', axis=1 , inplace=True)
ds.head()


# In[ ]:


scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )


# In[ ]:


scaled_ds.head()


# In[ ]:


scaled_ds.info()


# In[ ]:


scaled_ds.describe()


# In[ ]:


scaled_ds.head(10)


# In[ ]:


scaled_ds['NewIncome']=df.apply(lambda x: 1 if x['Income']>df.Income.median() else -1, axis=1)


# In[ ]:


scaled_ds.head(7)


# In[ ]:


scaled_ds["NewIncome"].value_counts()


# In[ ]:


scaled_ds["NewIncome"].fillna(df.mean(), inplace=True)


# In[ ]:


scaled_ds["NewIncome"].value_counts()


# In[ ]:


scaled_ds.shape


# In[ ]:


# Shuffle the data
scaled_ds = scaled_ds.sample(frac=1, random_state=15).reset_index(drop=True)
# Generate indices for splits
test_ind = round(len(df)*0.25)
train_ind = test_ind + round(len(df)*0.10)
unlabeled_ind = train_ind + round(len(df)*0.65)
# Partition the data
test = scaled_ds.iloc[:test_ind]
train = scaled_ds.iloc[test_ind:train_ind]
unlabeled = scaled_ds.iloc[train_ind:unlabeled_ind]


# In[ ]:


# Assign data to train, test, and unlabeled sets

X_train = train.drop('NewIncome', axis=1)
y_train = train.NewIncome.values

X_unlabeled = unlabeled.drop('NewIncome', axis=1)

X_test = test.drop('NewIncome', axis=1)
y_test = test.NewIncome.values


# Check dimensions of data after splitting

print(f"X_train dimensions: {X_train.shape}")
print(f"y_train dimensions: {y_train.shape}\n")

print(f"X_test dimensions: {X_test.shape}")
print(f"y_test dimensions: {y_test.shape}\n")

print(f"X_unlabeled dimensions: {X_unlabeled.shape}")

