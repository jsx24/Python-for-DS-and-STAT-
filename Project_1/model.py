#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


dfa=pd.read_csv(r'C:\Users\stpat\Desktop\avocado.csv', thousands=',')# data merged with PCI
dfb=pd.read_csv(r'C:\Users\stpat\Desktop\avocado1.csv')# data not merged PCI, original data 
dfa.head()


# In[2]:


# clean function 1 
def dfac(row):
    thishappy=row['region']
    try:thishappy=str(thishappy)
    except:pass
    
    try:
        if not isinstance(thishappy,str) : thishappy = str(thishappy)
    except: pass
  
    thishappy = thishappy.strip()
    if thishappy == "TotalUS": return 1
    if thishappy == "West": return 1
    if thishappy == "Southeast": return 1
    if thishappy == "Northeast": return 1
    if thishappy == "Midsouth": return 1
    if thishappy == "SouthCentral": return 1
    if thishappy == "NorthernNewEngland": return 1
    if thishappy == 'GreatLakes': return 1
    if thishappy == 'California': return 1
    if thishappy == 'SouthCarolina': return 1
    else: return thishappy
    
# subset 1 for first chart 
# this set only contain city 
sub1=dfb.copy()
sub1['region']=sub1.apply(dfac,axis=1)
sub1=sub1[sub1.region !=1]


# In[404]:


# first chart 
import plotly_express as px

fig1=px.scatter(sub1,x='Total Volume', y='AveragePrice', animation_frame='year',
          animation_group='region',color='region', size='Total Bags',hover_name='region',
          log_x=True,size_max=45,range_x=[20000,10000000],range_y=[0,4],
          title='Averge Price VS Total Volume\n from 2015-2018')
fig1.write_html(r"C:\Users\stpat\Desktop\plot.html")


# In[5]:


# clean function 2 for chart 2
def dfad(row):
    thishappy=row['region']
    try:thishappy=str(thishappy)
    except:pass
    
    try:
        if not isinstance(thishappy,str) : thishappy = str(thishappy)
    except: pass
  
    thishappy = thishappy.strip()
    if thishappy == "TotalUS": return thishappy
    if thishappy == "West": return thishappy
    if thishappy == "Southeast": return thishappy
    if thishappy == "Northeast": return thishappy
    if thishappy == "Midsouth": return thishappy
    if thishappy == "SouthCentral": return thishappy
    if thishappy == "NorthernNewEngland": return thishappy
    else: return 1

# subset 2 for chart 2
# this subset only contain region, not city 
sub2=dfa.copy()
sub2['region']=sub2.apply(dfad,axis=1)
sub2=sub2[sub2.region !=1]


# In[6]:


# chart 2
import seaborn as sns

sub2.sort_values('AveragePrice')
plt.figure(figsize=(20,10))
sns.barplot(x='year',y='AveragePrice', hue='region', data=sub2,
           palette='Blues_d')
plt.title('Average Price of Avocado in Different Region \n from 2015 to 2018', 
          fontsize=30)
plt.xticks(fontsize=25)
plt.xlabel('Year',fontsize=30)
plt.ylabel('Average Price',fontsize=30)
plt.yticks(fontsize=25)
plt.show()


# In[7]:


# clean function 3
# this function transfer city to state
def dfar(row):
    thishappy=row['region']
    try:thishappy=str(thishappy)
    except:pass
    
    try:
        if not isinstance(thishappy,str) : thishappy = str(thishappy)
    except: pass
  
    thishappy = thishappy.strip()
    if thishappy == "Albany": return 'NY'
    if thishappy == "Atlanta": return 'GA'
    if thishappy == "BaltimoreWashington": return 'MD'
    if thishappy == "Orlando": return 'FL'
    if thishappy == "Indianapolis": return 'MN'
    if thishappy == "HartfordSpringfield": return 'MA'
    if thishappy == "Louisville": return 'KY'
    if thishappy == "Chicago": return 'IL'
    if thishappy == "LasVegas ": return 'NV'
    if thishappy == "Nashville": return 'TN'
    if thishappy == "Tampa": return 'FL'
    if thishappy == "Sacramento": return 'CA'
    if thishappy == "Houston": return 'TX'
    if thishappy == "NewYork": return 'NY'
    if thishappy == "MiamiFtLauderdale": return 'FL'
    if thishappy == "Pittsburgh ": return 'PA'
    if thishappy == "SouthCarolina": return 'SC'
    if thishappy == "StLouis": return 'MS'
    if thishappy == "HarrisburgScranton": return 'PA'
    if thishappy == "RaleighGreensboro": return 'NC'
    if thishappy == "BuffaloRochester": return 'NY'
    if thishappy == "Boise": return 'ID'
    if thishappy == "Portland": return 'OR'
    if thishappy == "DallasFtWorth": return 'TX'
    if thishappy == "LosAngeles ": return 'CA'
    if thishappy == "SanFrancisco": return 'CA'
    if thishappy == "Boston": return 'MA'
    if thishappy == "Columbus": return 'OH'
    if thishappy == "California": return 'CA'
    if thishappy == "NewOrleansMobile": return 'LA'
    if thishappy == "SanDiego": return 'CA'
    if thishappy == "Charlotte": return 'NC'
    if thishappy == "Denver": return 'CO'
    if thishappy == "Spokane": return 'WA'
    if thishappy == "PhoenixTucson": return 'AZ'
    if thishappy == "Plains": return 'NE'
    if thishappy == "CincinnatiDayton": return 'OH'
    if thishappy == "Seattle": return 'WA'
    if thishappy == "Jacksonville": return 'FL'
    if thishappy == "Syracuse": return 'NY'
    if thishappy == "RichmondNorfolk": return 'VA'
    if thishappy == "Detroit": return 'MI'
    if thishappy == "Philadelphia": return 'PA'
    if thishappy == "GrandRapids": return 'MI'
    if thishappy == "WestTexNewMexico": return 'NM'
    else: return 1
    
# subset 3
# this subset change region to state 
sub3=dfa.copy()
sub3['region']=sub3.apply(dfar,axis=1)
sub3=sub3[sub3.region !=1]


# In[436]:


# clean function 4
# this function try to merge state PCI 
def dfat(row):
    thishappy=row['region']
    thisyear=row['year']
  
    thishappy = thishappy.strip()
    if thishappy == 'CA' and thisyear== 2015.0: return 55808
    if thishappy == 'CA' and thisyear== 2016.0: return 57801
    if thishappy == 'CA' and thisyear== 2017.0: return 60219
    if thishappy == 'CA' and thisyear== 2018.0: return 63711
    if thishappy == 'SC' and thisyear== 2015.0: return 39499
    if thishappy == 'SC' and thisyear== 2016.0: return 40406
    if thishappy == 'SC' and thisyear== 2017.0: return 42081
    if thishappy == 'SC' and thisyear== 2018.0: return 43702
    if thishappy == 'NY' and thisyear== 2015.0: return 57709
    if thishappy == 'NY' and thisyear== 2016.0: return 58856
    if thishappy == 'NY' and thisyear== 2017.0: return 60336
    if thishappy == 'NY' and thisyear== 2018.0: return 68710
    if thishappy == 'MD' and thisyear== 2015.0: return 57709
    if thishappy == 'MD' and thisyear== 2016.0: return 58856
    if thishappy == 'MD' and thisyear== 2017.0: return 60336
    if thishappy == 'MD' and thisyear== 2018.0: return 61112
    if thishappy == 'FL' and thisyear== 2015.0: return 41522
    if thishappy == 'FL' and thisyear== 2016.0: return 42047
    if thishappy == 'FL' and thisyear== 2017.0: return 42746
    if thishappy == 'FL' and thisyear== 2018.0: return 43535
    if thishappy == 'MN' and thisyear== 2015.0: return 43535
    if thishappy == 'MN' and thisyear== 2016.0: return 43535
    if thishappy == 'MN' and thisyear== 2017.0: return 43535
    if thishappy == 'MN' and thisyear== 2018.0: return 43535
    if thishappy == 'MA' and thisyear== 2015.0: return 63598
    if thishappy == 'MA' and thisyear== 2016.0: return 65496
    if thishappy == 'MA' and thisyear== 2017.0: return 68267
    if thishappy == 'MA' and thisyear== 2018.0: return 71886
    if thishappy == 'GA' and thisyear== 2015.0: return 41692
    if thishappy == 'GA' and thisyear== 2016.0: return 42705
    if thishappy == 'GA' and thisyear== 2017.0: return 44548
    if thishappy == 'GA' and thisyear== 2018.0: return 46519
    if thishappy == 'KY' and thisyear== 2015.0: return 39093
    if thishappy == 'KY' and thisyear== 2016.0: return 39638
    if thishappy == 'KY' and thisyear== 2017.0: return 41014
    if thishappy == 'KY' and thisyear== 2018.0: return 42527
    if thishappy == 'IL' and thisyear== 2015.0: return 51541
    if thishappy == 'IL' and thisyear== 2016.0: return 52299
    if thishappy == 'IL' and thisyear== 2017.0: return 53974
    if thishappy == 'IL' and thisyear== 2018.0: return 56919
    if thishappy == 'NV' and thisyear== 2015.0: return 44092
    if thishappy == 'NV' and thisyear== 2016.0: return 45001
    if thishappy == 'NV' and thisyear== 2017.0: return 46854
    if thishappy == 'NV' and thisyear== 2018.0: return 49290
    if thishappy == 'TN' and thisyear== 2015.0: return 42590
    if thishappy == 'TN' and thisyear== 2016.0: return 43720
    if thishappy == 'TN' and thisyear== 2017.0: return 44950
    if thishappy == 'TN' and thisyear== 2018.0: return 46889
    if thishappy == 'PA' and thisyear== 2015.0: return 50382
    if thishappy == 'PA' and thisyear== 2016.0: return 51619
    if thishappy == 'PA' and thisyear== 2017.0: return 53155
    if thishappy == 'PA' and thisyear== 2018.0: return 56252
    if thishappy == 'NM' and thisyear== 2015.0: return 43652
    if thishappy == 'NM' and thisyear== 2016.0: return 43635
    if thishappy == 'NM' and thisyear== 2017.0: return 43668
    if thishappy == 'NM' and thisyear== 2018.0: return 44728
    if thishappy == 'MI' and thisyear== 2015.0: return 43536
    if thishappy == 'MI' and thisyear== 2016.0: return 44874
    if thishappy == 'MI' and thisyear== 2017.0: return 46273
    if thishappy == 'MI' and thisyear== 2018.0: return 48480
    if thishappy == 'OH' and thisyear== 2015.0: return 44341
    if thishappy == 'OH' and thisyear== 2016.0: return 45053
    if thishappy == 'OH' and thisyear== 2017.0: return 46669
    if thishappy == 'OH' and thisyear== 2018.0: return 48793
    if thishappy == 'AZ' and thisyear== 2015.0: return 31019
    if thishappy == 'AZ' and thisyear== 2016.0: return 31798
    if thishappy == 'AZ' and thisyear== 2017.0: return 32397
    if thishappy == 'AZ' and thisyear== 2018.0: return 33476  
    if thishappy == 'WA' and thisyear== 2015.0: return 56354
    if thishappy == 'WA' and thisyear== 2016.0: return 57539
    if thishappy == 'WA' and thisyear== 2017.0: return 58918
    if thishappy == 'WA' and thisyear== 2018.0: return 60781
    if thishappy == 'CO' and thisyear== 2015.0: return 52147
    if thishappy == 'CO' and thisyear== 2016.0: return 52278
    if thishappy == 'CO' and thisyear== 2017.0: return 55374
    if thishappy == 'CO' and thisyear== 2018.0: return 58500
    if thishappy == 'ID' and thisyear== 2015.0: return 38286
    if thishappy == 'ID' and thisyear== 2016.0: return 38966
    if thishappy == 'ID' and thisyear== 2017.0: return 39032
    if thishappy == 'ID' and thisyear== 2018.0: return 40189
    if thishappy == 'NC' and thisyear== 2015.0: return 41857
    if thishappy == 'NC' and thisyear== 2016.0: return 42659
    if thishappy == 'NC' and thisyear== 2017.0: return 44192
    if thishappy == 'NC' and thisyear== 2018.0: return 46126
    if thishappy == 'NE' and thisyear== 2015.0: return 51565
    if thishappy == 'NE' and thisyear== 2016.0: return 53099
    if thishappy == 'NE' and thisyear== 2017.0: return 52878
    if thishappy == 'NE' and thisyear== 2018.0: return 53114  
    if thishappy == 'OR' and thisyear== 2015.0: return 45194
    if thishappy == 'OR' and thisyear== 2016.0: return 46514
    if thishappy == 'OR' and thisyear== 2017.0: return 48407
    if thishappy == 'OR' and thisyear== 2018.0: return 50951
    if thishappy == 'TX' and thisyear== 2015.0: return 46605
    if thishappy == 'TX' and thisyear== 2016.0: return 45654
    if thishappy == 'TX' and thisyear== 2017.0: return 47975
    if thishappy == 'TX' and thisyear== 2018.0: return 50483
    if thishappy == 'VA' and thisyear== 2015.0: return 52899
    if thishappy == 'VA' and thisyear== 2016.0: return 53611
    if thishappy == 'VA' and thisyear== 2017.0: return 55317
    if thishappy == 'VA' and thisyear== 2018.0: return 57910
    if thishappy == 'MS' and thisyear== 2015.0: return 35025
    if thishappy == 'MS' and thisyear== 2016.0: return 35618
    if thishappy == 'MS' and thisyear== 2017.0: return 36389
    if thishappy == 'MS' and thisyear== 2018.0: return 37904
    if thishappy == 'LA' and thisyear== 2015.0: return 47313
    if thishappy == 'LA' and thisyear== 2016.0: return 46117
    if thishappy == 'LA' and thisyear== 2017.0: return 46145
    if thishappy == 'LA' and thisyear== 2018.0: return 46120
    else: return 1

# subset 4 with merged state PCI and region become state 
sub4=sub3.copy()
sub4['PCI']=sub4.apply(dfat,axis=1)


# In[10]:


sub4.head()


# In[11]:


# map chart 
from urllib.request import urlopen
import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[437]:


# subset 4.1 for map chart 
# this subset only have 2015 data 
sub4_1=sub4.loc[sub4['year'] == 2015]


# In[438]:


# map chart 

import plotly.graph_objects as go

for col in sub4_1.columns:
    sub4_1[col] = sub4_1[col].astype(str)
sub4_1['text'] = sub4_1['region'] + '<br>' +     'PCI: $' + sub4_1['PCI'] 


# In[439]:



fig = go.Figure(data=go.Choropleth(
    locations=sub4_1['region'],
    z=sub4_1['AveragePrice'].astype(float),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=False,
    text=sub4_1['text'], 
    marker_line_color='white', 
    colorbar_title="USD"
))

fig.update_layout(
    title_text='Average Price and Per Capital Personal Income <br>(2015)',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
)

fig.show()
fig.write_html(r"C:\Users\stpat\Desktop\mapplot2.html")


# In[15]:


dfb['Total Volume'].describe()


# In[16]:


# clean function 5 
# this function creat the variable 'volume level' 
def dfa5(row):
    a=row['Total Volume']
    
    try:a=float(a)
    except:pass
    
    try:
        if not isinstance(a,float) : thishappy = float(thishappy)
    except: pass
     
    if a > 432962.8: return'high'
    if a <10838.7: return'low'
    else: return'medium'


# In[393]:


sub5=dfa.copy()
sub5['volume level']=sub5.apply(dfa5,axis=1)
sub5.head()


# In[347]:


# linear regression model 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from statsmodels.formula.api import ols


# In[394]:


sub5_1=pd.get_dummies(sub5, columns=['type','region', 'volume level'])


# In[395]:


sub5_1.head()


# In[396]:


sub5_1['PCI']=sub5_1["PCI"].replace('            NA',np.NaN)


# In[397]:


sub5_1=sub5_1.dropna()


# In[398]:


sub5_1["PCI"] = sub5_1["PCI"].str.replace(",","").astype(float)


# In[399]:


numeric_feats = sub5_1.dtypes[sub5_1.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = sub5_1[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[400]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))


skewed_features = skewness.index
lam=0.15
for feat in skewed_features:
    sub5_1[feat] = boxcox1p(sub5_1[feat], lam)


# In[401]:


sns.distplot(sub5_1['AveragePrice'], fit=norm)
(mu, sigma) = norm.fit(sub5_1['AveragePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('AveragePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(sub5_1['AveragePrice'], plot=plt)
plt.show()


# In[273]:


cor=sub5_1.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(cor, cmap="YlGnBu")


# In[280]:


print(sm.graphics.tsa.acf(at, nlags=40))
sm.graphics.tsa.plot_pacf(at, lags=40)
plt.show()


# In[368]:



x=sub5_1.iloc[:,3:]
y=sub5_1['AveragePrice']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=2020)


# In[369]:


M1 = Pipeline([('sc', StandardScaler()),
               ('pca', PCA(n_components=0.95)),
               ('ols', LinearRegression())
                    ])

M1.fit(X_train, y_train)
plt.plot(X_train, y_train, 'o')
plt.plot(X_test, M1.predict(X_test))


# In[294]:


print(M1.named_steps['ols'].intercept_)
print(M1.named_steps['ols'].coef_)


# In[327]:


M2 = Pipeline([('sc', StandardScaler()),
               ('pca', PCA(n_components=0.95)),
               ('GBoost',GradientBoostingRegressor())
                    ])


parameters1 = {
    "GBoost__loss":["huber"],
    "GBoost__min_samples_split": np.linspace(0.1, 0.5, 12),
    "GBoost__min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "GBoost__max_depth":[3,5,8],
    "GBoost__max_features":["log2","sqrt"],
    "GBoost__subsample":[0.6, 0.8, 1.0],
    }


GBoost_grid = GridSearchCV(M2,
                        parameters1,
                        cv = 10,
                        n_jobs = -1,
                        verbose=True)


get_ipython().run_line_magic('time', '_=GBoost_grid.fit(X_train, y_train)')

print ('Best Parameter \n', GBoost_grid.best_params_, '\n Best Score: %.3f' % GBoost_grid.best_score_)


# In[ ]:


params = {
    'loss': ["huber"],
    'min_samples_leaf': 0.5,
    'max_depth': 8,
    'min_sample_split': 0.5,
    'max_features': ["sqrt"],
    'maxsample': 0.8
    
}

bst1 = GradientBoostingRegressor.fit(param) 
bst1.fit(X_train, y_train)
print ('Score: %.3f' %bst1.score(X_test,y_test))


# In[371]:


M3 = Pipeline([('sc', StandardScaler()),
               ('pca', PCA(n_components=0.95)),
               ('XGB',xgb.XGBRegressor())
                    ])

parameters2 = {'XGB__nthread':[-1], 
              'XGB__objective':['reg:linear'],
              'XGB__learning_rate': [.03, 0.05, .07], 
              'XGB__max_depth': [5, 6, 7],
              'XGB__min_child_weight': [4],
              'XGB__silent': [1],
              'XGB__subsample': [0.6, 0.8, 1.0],
              'XGB__colsample_bytree': [0.6, 0.8, 1.0]}



XGB_grid = GridSearchCV(M3,
                        parameters2,
                        cv = 2,
                        n_jobs = -1,
                        verbose=True)


get_ipython().run_line_magic('time', '_=XGB_grid.fit(X_train, y_train)')

print ('Best Parameter: \n', XGB_grid.best_params_, '\nBest Score:%.3f'% XGB_grid.best_score_)


# In[ ]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


def huber_approx_obj(preds, dtrain):
    d = preds - dtrain.get_labels() #remove .get_labels() for sklearn
    h = 1  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess

params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.07,
    'max_depth': 7,
    'min_child_weight': 4,
    'nthread': -1,
    'silent': 1,
    'subsample': 0.8
    
}

bst2 = xgb.XGBRegressor(param,obj=huber_approx_obj) 
bst2.fit(dtrain)
print ('Score: %.3f' %bst2.score(dtest))


# In[361]:


M4 = Pipeline([('sc', StandardScaler()),
               ('pca', PCA(n_components=0.95)),
               ('LGBM',lgb.LGBMRegressor())
                    ])


parameters3 = {'LGBM__learning_rate':[0.01, 0.05, 0.1, 1],
               'LGBM__n_estimators':[500,720,1000],
               'LGBM__num_leaves':[15,31,60],
              }



LGBM_grid = GridSearchCV(M4,
                        parameters3,
                        cv = 2,
                        n_jobs = -1)


get_ipython().run_line_magic('time', '_=LGBM_grid.fit(X_train, y_train)')

print ('Best Parameter \n', LGBM_grid.best_params_, '\nBest Score: %.3f'%LGBM_grid.best_score_)



# In[ ]:


def smape_loss(preds, train_data):
    print('************************************')
    print(preds)
    labels = train_data.get_label()
    grad = np.zeros(shape=len(preds), dtype=np.float64)
    for x in range(len(preds)):
        if preds[x] >= labels[x]:
            grad[x] = 1 / labels[x]
        else:
            grad[x] = - 1 / labels[x]
    hess = np.zeros(shape=len(preds), dtype=np.float64)
    print(grad)
    print(hess)
    return grad, hess


params = {
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'num_leaves': 60
}

bst3 = lgb(params,fobj=smape_loss, feval=smape, valid_sets=[valid_set], verbose_eval=10, early_stopping_rounds=100)

print ('Score: %.3f' %bst3.score(X_test,y_test))


# In[380]:


testdata=sub5_1.loc[(sub5_1['year']==2018.0)&(sub5_1['region_Atlanta']==1)&(sub5_1['Total Volume']==15714.11)]


# In[362]:


X_train['Total Volume'].head()


# In[472]:


t1=pd.read_csv(r'C:\Users\stpat\Desktop\testset.csv')


# In[473]:


t1['volume level']=t1.apply(dfa5,axis=1)


# In[474]:


t2=pd.get_dummies(t1, columns=['region','type','volume level'])
t2.head()


# In[476]:


t2['PCI']=t2["PCI"].replace('            NA',np.NaN)


# In[477]:


t2=t2.dropna()


# In[478]:


t2["PCI"] = t2["PCI"].str.replace(",","").astype(float)


# In[479]:


X_t=t2.drop('AveragePrice', 1)
Y_t=t2['AveragePrice']


# In[492]:


tt=ac.head(32)
cc=Y_t.head(32)
ff=cc-tt
gg=ff.iloc[:,0]
print(gg)


# In[3]:


import plotly.graph_objects as go

headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

fig3 = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Features</b>','<b>GBoost</b>','<b>XGBoost</b>','<b>LightGBM </b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='white', size=12)
  ),
  cells=dict(
    values=[
      ['Time', 'Huber Loss', 'Best Parameter'],
      ['31.2 min', 0.67, 'min_sample_split:0.5, max_features: sqrt '],
      ['13.36 min', 0.82, 'colsample_bytree: 1.0, XGB__learning_rate: 0.07'],
      ['17.27 min', 0.87, 'learning_rate=0.1, n_estimators=1000']],
    line_color = 'darkslategray',
    # 2-D list of colors for alternating rows
    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor]*3],
    align = ['left', 'center'],
    font = dict(color = 'darkslategray', size = 11)
    ))
])

fig3.show()


# In[ ]:




