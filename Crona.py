# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:31:19 2020

@author: japesh
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from keras.callbacks import ModelCheckpoint
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from fancyimpute import  SoftImpute
from sklearn import preprocessing
import datetime
import tensorflow as tf
import pandas_profiling
from keras.layers import Dropout


df=pd.read_csv('Train_dataset1.csv')

#profile = df.profile_report(title='Pandas Profiling Report')
#profile.to_file(output_file="Pandas Profiling Report — AirBNB .html")

df_final=df[['Infect_Prob']]

head=df.head()
oh1=pd.get_dummies(df['Gender'])
oh1.drop(columns=['Female'],inplace=True)

oh3=pd.get_dummies(df['Married'])
oh3.drop(columns=['NO'],inplace=True)

oh4=pd.get_dummies(df['Children'])
oh4.drop(columns=[2.0],inplace=True)

oh5=pd.get_dummies(df['Occupation'])
oh5.drop(columns=['Farmer'],inplace=True)

oh6=pd.get_dummies(df['Mode_transport'])
oh6.drop(columns=['Walk'],inplace=True)

oh7=pd.get_dummies(df['comorbidity'])
oh7.drop(columns=['None'],inplace=True)

oh8=pd.get_dummies(df['Pulmonary score'])
oh8.drop(columns=['<400'],inplace=True)

oh9=pd.get_dummies(df['cardiological pressure'])
oh9.drop(columns=['Normal'],inplace=True)

df.drop(columns=['people_ID','Region','Gender','Designation','Name','Married','Children','Occupation','Mode_transport','comorbidity','Pulmonary score','cardiological pressure','Infect_Prob'],inplace=True)



df=df.join(oh1).join(oh3).join(oh4).join(oh5).join(oh6).join(oh7).join(oh8).join(oh9).join(df_final)


df.isnull().sum().sort_values(ascending=False)


x = df.iloc[:,:-1].values #returns a numpy array
y=df.iloc[:,-1].values
y=np.reshape(y, (-1,1))

min_max_scaler = preprocessing.MinMaxScaler()
scaler_y = preprocessing.MinMaxScaler()
y=scaler_y.fit_transform(y)
x_scaled = min_max_scaler.fit_transform(x)
df1 = pd.DataFrame(x_scaled)




X_filled_softimpute = SoftImpute().fit_transform(df1)
df_filled = pd.DataFrame(X_filled_softimpute)


df_filled.isnull().sum().sort_values(ascending=False)



df_filled.columns = [              'cases/1M',              'Deaths/1M',
                          'Age',             'Coma score',
                     'Diuresis',              'Platelets',
                          'HBB',                'd-dimer',
                   'Heart rate',        'HDL cholesterol',
               'Charlson Index',          'Blood Glucose',
                    'Insurance',                 'salary',
                     'FT/month',                   'Male',
                          'YES',                      '0.0',
                            '1.0',               'Business',
                      'Cleaner',                  'Clerk',
                       'Driver',                  'Legal',
                'Manufacturing',             'Researcher',
                        'Sales',                    'Car',
                       'Public', 'Coronary Heart Disease',
                     'Diabetes',           'Hypertension',
                         '<100',                   '<200',
                         '<300',               'Elevated',
                     'Stage-01',               'Stage-02']

#profile = df_filled.profile_report(title='Pandas Profiling Report')
#profile.to_file(output_file="Pandas Profiling Report — corona.html")

X=df_filled

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X)


#
#
#
#classifier = Sequential()
#
#classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 38))
#classifier.add(Dropout(0.3))
#classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(0.3))
#classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
#classifier.summary()
#
##log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
##tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
#
#history=classifier.fit(X_train, y_train, epochs = 10,batch_size=32,validation_data=(X_test, y_test))
#
#
##print(history.history.keys())
## "Loss"
#yini=classifier.predict(X_train)
#
#yini = scaler_y.inverse_transform(yini)
#
#
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'])
#plt.show()
#
#
#
#
##### load or save  model####
#classifier.save('classifier2_final.h5')


deep = load_model('classifier1_final.h5')



dfnew=pd.read_csv('Test_dataset1.csv')
df_ID=dfnew[['people_ID']]



head=dfnew.head()
oh1=pd.get_dummies(dfnew['Gender'])
oh1.drop(columns=['Female'],inplace=True)


oh3=pd.get_dummies(dfnew['Married'])
oh3.drop(columns=['NO'],inplace=True)

oh4=pd.get_dummies(dfnew['Children'])
oh4.drop(columns=[2.0],inplace=True)

oh5=pd.get_dummies(dfnew['Occupation'])
oh5.drop(columns=['Farmer'],inplace=True)

oh6=pd.get_dummies(dfnew['Mode_transport'])
oh6.drop(columns=['Walk'],inplace=True)

oh7=pd.get_dummies(dfnew['comorbidity'])
oh7.drop(columns=['None'],inplace=True)

oh8=pd.get_dummies(dfnew['Pulmonary score'])
oh8.drop(columns=['<400'],inplace=True)

oh9=pd.get_dummies(dfnew['cardiological pressure'])
oh9.drop(columns=['Normal'],inplace=True)

dfnew.drop(columns=['people_ID','Region','Gender','Designation','Name','Married','Children','Occupation','Mode_transport','comorbidity','Pulmonary score','cardiological pressure'],inplace=True)

dfnew=dfnew.join(oh1).join(oh3).join(oh4).join(oh5).join(oh6).join(oh7).join(oh8).join(oh9)

X_result=dfnew.values
X_result=min_max_scaler.transform(X_result)


ynew=regressor.predict(X_result)
ynew=np.reshape(ynew, (-1,1))

ynew = scaler_y.inverse_transform(ynew)


ynew1=regressor.predict(X)
ynew1=np.reshape(ynew, (-1,1))

ynew1 = scaler_y.inverse_transform(ynew1)

ydeep=deep.predict(X_result)
ydeep=np.reshape(ydeep, (-1,1))
ydeep = scaler_y.inverse_transform(ydeep)

ans=df_ID.join(pd.DataFrame(ydeep))


ans.to_csv('file1.csv') 
