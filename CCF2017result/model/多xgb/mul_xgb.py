# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 10:34:02 2017

@author: Administrator
"""

import pandas as pd
import xgboost as xgb

data_path = "C:/Luoshichao/ccf_4/quarter-final/input/"
all_data = pd.read_hdf(data_path+'processedData/all_data_1.h5')

feature_ex=pd.read_csv('C:/Luoshichao/ccf_4/quarter-final/input/processedData/feature_ex.csv')
feature_ex.drop('EID',axis=1,inplace=True)

all_data=pd.concat([all_data,feature_ex],axis=1)
del feature_ex

# dealing EID
all_data = all_data.assign(char_EID = lambda x:x.EID.map(lambda x:x[:1] if type(x) == str else 't')) \
    .assign(num_EID = lambda x:x.EID.map(lambda x:x[1:] if type(x) == str else x)) \
    .assign(num_EID = lambda x:x.num_EID.astype(float)) \
    .drop('char_EID',1)


train_data = all_data[all_data.TARGET.notnull()]
test_data = all_data[all_data.TARGET.isnull()]

train_eid=train_data.pop('EID')
EID = test_data.pop('EID')

y_train = train_data.pop('TARGET')
test_data.pop('TARGET')
train_data.pop('ENDDATE')
test_data.pop('ENDDATE')



train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

a=pd.read_csv('C:/Luoshichao/gbdt/a.csv')
b=pd.read_csv('C:/Luoshichao/gbdt/b.csv')
a=a.set_index('Unnamed: 0')
b=b.set_index('Unnamed: 0')
train_data=pd.concat([train_data,a],axis=1)
test_data=pd.concat([test_data,b],axis=1)
del all_data

test_result = pd.DataFrame(EID.values,columns=["EID"])
depth=[6,7,8]
nround=[2250,2000,1750]
for i in range(10):
    model_xgb = xgb.XGBClassifier(
        learning_rate=0.02,
        n_estimators=nround[i%3],
        min_child_weight=2,
        max_depth=depth[i%3],
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.6,
        objective='binary:logistic',
        nthread=12,
        scale_pos_weight=1,
        seed=27*i)
    model_xgb.fit(train_data,y_train)
    test_y_prob = model_xgb.predict_proba(test_data)[:,1]  
    test_result['PROB_'+str(i)] = test_y_prob
test_result.to_csv('C:/Luoshichao/result/test_result.csv',index=None,encoding='utf-8')
result=test_result
scr_test=result[['PROB_'+str(i) for i in range(10)]].mean(axis=1)
result['PROB']=scr_test
df=result[['EID','PROB']]
df['FORTARGET']=df['PROB'].apply(lambda x: 1 if x>0.25 else 0)
df1=df[['EID','FORTARGET','PROB']]
df1.to_csv('C:/Luoshichao/result/mul_xgb.csv',index=None,encoding='utf-8')
    







