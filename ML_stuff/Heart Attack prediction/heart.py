# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# %%
df = pd.read_csv('data/heart.csv')

# %%
df.info()

# %%
df.describe()

# %%
#Lookin at missing values 
#There are no missing values
df.isna().sum()

# %%
df.head()

# %% [markdown]
# ## EDA
# 

# %%
#Taking a look at correlation matrix
plt.figure(figsize = (13,10))
sns.heatmap(df.corr() , annot= True )

# %%
#Starting from age column 
df.age.iloc[:5]

# %%
plt.figure(figsize = (14,7))
sns.countplot(data = df , x = 'age', hue = 'output')

# %%
sns.histplot(data = df , x = 'age' , kde = True)
print("The skewness of Age column is {}".format(df.age.skew()))

# %% [markdown]
# It seems like after a certain age the chance of heart attack has decereased .
# Also the data seems to be close to normally distributed which a good sign 

# %%
sns.boxplot(data = df , y = 'age' )
#seems have to no outliers 

# %%
#Age groups 
#30 - 45 - 1 
#46 - 60 - 2 
#60+ - 3
df['Age group'] = df['age'].apply(lambda x : 1 if 29 <= x <= 45 else( 2 if 46 <= x <= 60 else 3))

# %%
df['Age group'].iloc[:5]

# %%
sns.countplot(data = df , x = 'Age group' , hue = 'output' )

# %%
df['age'] = np.log(df['age'].values)

# %%
#Taking a look sex column now 
sns.countplot(data = df , x = 'sex' , hue = 'output')
plt.xlabel('Gender')

# %%
female = df[(df['sex']  == 0) & (df['output'] == 1)]
total_female = df[df['sex'] == 0]
pct_female = (len(female) / len(total_female)) * 100
male = df[(df['sex'] == 1) &  (df['output'] == 1)]
total_male = df[df['sex'] == 1]
pct_male = (len(male) / len(total_male)) * 100
plt.pie(x = [pct_female,pct_male] , labels = ['Female' ,'Male'] , shadow=True , explode=[0.35,0]  , autopct='%.2f')


# %% [markdown]
# Looks like women had higher chances of heart attack . so women are more likely to have higher chances of heart attack

# %%
#Looking at chest pain column 
sns.countplot(data = df , x = 'cp' , hue = 'output')

# %% [markdown]
# The asymptotic pain is a type of chest pain which cant meet the criteria and from the above plot it seems its very less lethal compared to other chest pains 

# %% [markdown]
# **trtbps** is the resting blood pressure of the person . A normal resting blood pressure should be under 120 .

# %%
sns.histplot(data = df , x = 'trtbps' , kde = True )

# %%
danger = len(df[df['trtbps'] > 120])
print("The number of people with resting blood pressure above 120 are {}".format(danger))

# %%
danger_1 = len(df[(df['trtbps'] > 120 ) & (df['output'] == 1)])
pct_danger = (danger_1 / danger) * 100
plt.pie(x = [danger_1,(danger - danger_1)] , labels = ['People above 120 with chance of heart attack' ,'Less chance of heart attack and above 120'] , shadow=True , explode=[0.35,0]  , autopct='%.2f')


# %%
normal = len(df[df['trtbps'] <= 120])
normal_1 = len(df[(df['trtbps'] <= 120) & (df['trtbps'] == 1)])
print("People with BP under 120 and have high chance of heart attack {}".format(normal_1))

# %% [markdown]
# So, if you have a normal BP then the danger of heart attack seems to be 0 , this is a really good piece of information

# %%
df['Danger_BP'] = df['trtbps'].apply(lambda x : 1 if 120 < x < 141 else (2 if x >= 141 else 0))

# %%
df['trtbps'] = np.log(df['trtbps'].values)

# %%
sns.histplot(data = df , x = 'trtbps' , kde = True )

# %% [markdown]
# **chol** seems to be related to cholestrol and this column may give us some good insights.

# %%
sns.histplot(data = df , x = 'chol' , kde = True , hue = 'output')

# %%
plt.scatter(x = df['chol'] , y = df['output'])

# %%
#Applying log transformation will reduce skewness
df['chol'] = np.log(df['chol'].values)

# %%
sns.histplot(data = df , x = 'chol' , kde = True , hue = 'output')

# %% [markdown]
# **fbs** is related to fasting blood sugar levels . An healthy persons level should be between 70 mg/dL (3.9 mmol/L) and 100 mg/dL (5.6 mmol/L). if the fbs is above 120 than in the column its given as 1 

# %%
sns.countplot(data = df , x  = 'fbs' , hue = 'output')

# %% [markdown]
# Looks like having free blood sugar level less than 120 still has higher chance of heart attack

# %% [markdown]
# **restecg** resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy)

# %%
sns.histplot(data = df , x = 'restecg' , hue = 'output' , kde = True)

# %% [markdown]
# So , having ST-T wave abnormality has very high chance of heart attack

# %% [markdown]
# **thalach** - Maximum heart beat reached , the maximum heart reached for any person should be age subtracted from 220

# %%
sns.histplot(data = df , x = 'thalachh' , kde = True , hue = 'output')

# %%
sns.boxplot(data = df , y = 'thalachh')

# %% [markdown]
# The heart attack chances greatly increase after the highest heart beat reaches above 150 approx

# %%
df['MAX_HR'] = df['thalachh'].apply(lambda x : 1 if x > 150 else 0)

# %% [markdown]
# **execg** - refers to excercised induced aningna , i suppose it refers to chest pain that is induced during exercise

# %%
sns.countplot(data = df , x = 'exng' , hue = 'output')

# %% [markdown]
# So, it seems the anigna induced by exercises has very less likely of a heart attack compared to non-exercise anigna 

# %% [markdown]
# **oldpeak** - ST depression induced by exercise relative to rest

# %%
sns.histplot(data = df , x = 'oldpeak' , hue = 'output' , kde = True)

# %% [markdown]
# **slope** - the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)

# %%
sns.countplot(data = df , x = 'slp' , hue = 'output')

# %% [markdown]
# having a down slope seems to have high risk of heart attack

# %% [markdown]
# **ca** - number of major vessels (0-3) colored by flourosopy
# 

# %%
sns.countplot(data = df , x = 'caa' , hue = 'output')

# %%
df['caa'] = df['caa'].replace(to_replace=4,value=3)

# %%
print(df.caa)

# %% [markdown]
# Having no major vessel puts the person at very high risk of heart attack

# %%
df['no_mv'] = df['caa'].apply(lambda x : 1 if x == 0 else 0)

# %%
print(df['no_mv'].value_counts())

# %% [markdown]
# **thal** - 2 = normal; 1 = fixed defect; 3 = reversable defect

# %%
sns.countplot(data = df , x = 'thall' , hue = 'output')

# %%
df['thall_2'] = df['thall'].apply(lambda x : 1 if x == 2 else 0)

# %%
plt.figure(figsize=(14,10))
sns.heatmap(df.corr() , annot=True)

# %%
df.drop(columns = ['MAX_HR','no_mv','thall_2'] ,axis = 1 , inplace = True)

# %% [markdown]
# ## Modelling

# %%
df.head()

# %%
y = df['output']
df.drop(columns = ['output'] , axis = 1,inplace=True)
X = df
y =  np.array(y).reshape(-1,1)
col = df.columns

# %%
print(X.shape)
print(y.shape)

# %%
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y ,random_state=0, test_size=0.2)

# %%
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# %%


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

models = {
    "LR" : LogisticRegression(),
    "SVC" : SVC(),
    "DT" : DecisionTreeClassifier(),
    "RF" : RandomForestClassifier(),
    'XG' : XGBClassifier()
}

names = []
train_score = []
valid_score = []

for name , model in models.items():
    names.append(name)
    model.fit(X_train,y_train)
    train_score.append(accuracy_score(y_train,model.predict(X_train)))
    valid_score.append(accuracy_score(y_test,model.predict(X_test)))



# %%
print(names)
print(train_score)
print(valid_score)

# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier().fit(X_train,y_train)
print(accuracy_score(y_train,knn.predict(X_train)))
print(accuracy_score(y_test,knn.predict(X_test)))

# %%
import optuna
def objective(trial):
    param = {
        "kernel" : trial.suggest_categorical("kernel",["linear","poly","rbf"]),
        "C" : trial.suggest_float("C",0,100.0),
        "degree" : trial.suggest_int("degree",1,100),
        
    }

    model = SVC(**param,random_state=0).fit(X_train,y_train)
    accuracy = accuracy_score(y_test,model.predict(X_test))
    return accuracy

study = optuna.create_study(direction = 'maximize')
study.optimize(objective , n_trials= 100)

# %%
print(study.best_params)

# %%
svc = SVC(**study.best_params).fit(X_train,y_train)
print("The train accuracy is {}".format(accuracy_score(y_train,svc.predict(X_train))))
print("The test accuracy is {}".format(accuracy_score(y_test,svc.predict(X_test))))


