import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, cross_val_score

############################# separate data ##############################

data_path = "../../input/"
all_data = pd.read_hdf(data_path+'processedData/all_data.h5')

# dealing EID
all_data = all_data.assign(char_EID = lambda x:x.EID.map(lambda x:x[:1] if type(x) == str else 't')) \
    .assign(num_EID = lambda x:x.EID.map(lambda x:x[1:] if type(x) == str else x)) \
    .assign(num_EID = lambda x:x.num_EID.astype(float)) \
    .drop('char_EID',1)

all_data.pop('ALTDIFF')

train_data = all_data[all_data.TARGET.notnull()]
test_data = all_data[all_data.TARGET.isnull()]
train_data.pop('EID')
EID = test_data.pop('EID')

y_train = train_data.pop('TARGET')
test_data.pop('TARGET')
train_data.pop('ENDDATE')
test_data.pop('ENDDATE')

################################添加xgb特征##########################################
# xgb_train_feature = pd.read_hdf(data_path + 'processedData/xgb_train_feature.h5')
# xgb_test_feature = pd.read_hdf(data_path + 'processedData/xgb_test_feature.h5')
# cols = range(0,100)
# xgb_train = xgb_train_feature[xgb_train_feature.columns[:500]]
# xgb_test = xgb_test_feature[xgb_test_feature.columns[:500]]
# xgb_test.index = test_data.index
# train_data = pd.concat([train_data,xgb_train],axis = 1)
# test_data = pd.concat([test_data,xgb_test],axis = 1)
#################################添加lgb特征#####################################################
# a=pd.read_csv(data_path + 'processedData/a.csv')
# b=pd.read_csv(data_path + 'processedData/b.csv')
# a=a.set_index('Unnamed: 0')
# b=b.set_index('Unnamed: 0')
# train_data=pd.concat([train_data,a],axis=1)
# test_data=pd.concat([test_data,b],axis=1)
# del a
# del b
##########################################################################################

train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
print(train_data)
print(test_data)

##############################model####################################
def rmsle_cv(model,train_data,label):
    rmse = cross_val_score(model, train_data, label,
                                    scoring="roc_auc", cv = 3,verbose=5) # auc评价
    return(rmse.mean())

def scale(pred_data):
    max = pred_data.max()
    min = pred_data.min()
    scale_preds = []
    for pred in pred_data:
        scale_pred = (pred-min)/(max-min)
        scale_preds.append(scale_pred)
    scale_preds = pd.Series(scale_preds)
    return scale_preds

rf_param_grid = {
    # 'max_depth' : [30,40,50,60,100],
    # 'min_samples_split' : [50,80,100,120],
    # 'min_samples_leaf' : [8,10,12],
    'max_features' : [0.04,0.05,0.06]
    # 'min_weight_fraction_leaf' : [1e-5,1e-3,0.01,0.1]
    # 'max_leaf_nodes' : [1000,2000]
    # 'n_estimators' : [500,800,1000,2000],
    # 'max_depth' : [50,80,120]
}

rf = RandomForestRegressor(
    n_estimators=800,max_depth=80,oob_score=True, # n_estimators=500,
    min_samples_split=80,max_features='sqrt',
    min_samples_leaf=10,random_state=5, # min_samples_leaf=10
    n_jobs=4,
)

# gsrf = GridSearchCV(rf,rf_param_grid,cv=3,scoring='roc_auc',verbose=5,return_train_score = True)
#
# gsrf.fit(train_data,y_train)
# rf_best = gsrf.best_estimator_
# print(rf_best)
# # Best score
# print(gsrf.best_score_)
#
# auc = rmsle_cv(rf,train_data,y_train)
# print('rf AUC : ',auc)

rf.fit(train_data,y_train)
y_hat = rf.predict(test_data)
y_hat = scale(y_hat)
y_prob = []
for y in y_hat:
    y_prob.append(y)
y_hat[y_hat > 0.18] = 1
y_hat[~(y_hat > 0.18)] = 0
#
submission_data = pd.DataFrame(data= {'EID':EID.values,
                                      'FORTARGET': y_hat,
                                      'PROB':y_prob
                                      },
                               columns=['EID','FORTARGET','PROB'])
submission_data.to_csv(path_or_buf='../../output/rf.csv',index=None,header=True)