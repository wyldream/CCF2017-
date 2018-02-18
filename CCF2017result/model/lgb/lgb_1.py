from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import bagging
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression,Lasso,Ridge
from sklearn.model_selection import cross_val_score
import sklearn.preprocessing as preprocessing
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split,KFold
from sklearn.learning_curve import validation_curve
import numpy as np
import pandas as pd

############################# separate data ##############################

data_path = "C:/Luoshichao/ccf_4/quarter-final/input/"
all_data = pd.read_hdf(data_path+'processedData/all_data.h5')

# dealing EID
all_data = all_data.assign(char_EID = lambda x:x.EID.map(lambda x:x[:1] if type(x) == str else 't')) \
    .assign(num_EID = lambda x:x.EID.map(lambda x:x[1:] if type(x) == str else x)) \
    .assign(num_EID = lambda x:x.num_EID.astype(float)) \
    .drop('char_EID',1)


train_data = all_data[all_data.TARGET.notnull()]
test_data = all_data[all_data.TARGET.isnull()]
train_data.pop('EID')
EID = test_data.pop('EID')

y_train = train_data.pop('TARGET')
test_data.pop('TARGET')
train_data.pop('ENDDATE')
test_data.pop('ENDDATE')



train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

def rmsle_cv(model,train_data,label):
    rmse = cross_val_score(model, train_data, label,scoring="roc_auc", cv = 3,verbose=5) # auc评价

def scale(pred_data):
    max = pred_data.max()
    min = pred_data.min()
    scale_preds = []
    for pred in pred_data:
        scale_pred = (pred-min)/(max-min)
        scale_preds.append(scale_pred)
    scale_preds = pd.Series(scale_preds)
    return scale_preds

lgb_params = {
    # 'max_bin' : [100,200,300,500],
    # 'num_leaves' : [200,150,100,],
    # 'min_sum_hessian_in_leaf' : [40,50,60]
    # 'max_depth' : [3,5,7,9]
    # 'learning_rate' : [0.001,0.01,0.02,0.05],
    # 'n_estimators' : [1000,1500,2000,3000]
    # 'min_data_in_leaf' :[4,6,8,10,20],
    # 'min_gain_to_split' : [0.01,0.1,0.5]
    'subsample' : [0.7,0.8,0.9],
    'colsample_bytree' : [0.5,0.6,0.7]
    # 'lambda_l1' : [0.01,0.1,0.5],
    # 'lambda_l2' : [0.001,0.01,0.1]
}

'''
model_lgb =LGBMClassifier(boosting_type='gbdt',
                     objective='binary',  # objective='multiclass', num_class = 3【多分类要指定类别数】
                     max_depth=-1,
                     num_leaves=2**8,
                     learning_rate=0.01,
                     n_estimators=2000,
                     min_split_gain=0.0,
                     min_child_weight=5,
                     subsample=0.88,
                     subsample_freq=1,
                     colsample_bytree=0.88,
                     reg_alpha=10.0,
                     reg_lambda=10.0,
                     scale_pos_weight=1,
                     seed=1024,
                     nthread=12)
'''





# LGBR = GridSearchCV(model_lgb,lgb_params,cv=3,scoring='roc_auc',verbose=5,return_train_score = True)
# LGBR.fit(train_data,y_train)
# LGB_best = LGBR.best_estimator_
# print(LGB_best)
# print(LGBR.best_score_)

model_lgb = lgb.LGBMRegressor(boosting_type='gbdt',
                              objective='regression',
                              learning_rate=0.01, n_estimators=3000,
                              reg_alpha = 0.1,reg_lambda = 0.01,
                              max_bin = 200,  num_leaves=150,max_depth=-1,
                              subsample_freq=5, colsample_bytree = 0.6,subsample = 0.8, 
                              min_child_samples=50,min_split_gain=0,seed=1024,nthread=-1)
#lgb_1

auc = rmsle_cv(model_lgb,train_data,y_train)
print('model_lgb AUC : ',auc)

model_lgb.fit(train_data,y_train)
test_y_prob = model_lgb.predict(test_data)
test_y_prob = scale(test_y_prob)
test_y_cat=[int(item>0.25) for item in list(test_y_prob)]
test_result = pd.DataFrame(EID.values,columns=["EID"])
test_result['FORTARGET']=test_y_cat
test_result["PROB"] = test_y_prob.values
test_result.to_csv('C:/Luoshichao/ccf_4/result/lgb_1.csv',index=None,encoding='utf-8')