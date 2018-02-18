from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC ,Ridge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import numpy as np

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
######################################################################################

train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
print(train_data)
print(test_data)

# 判断inf
# inf_pos = np.where(np.isinf(train_data))
# print(inf_pos)
# print(np.where(np.isinf(train_data.ix[128])))
# print(train_data.ix[152247][120:135]) # ALTDIFF
# print(np.where(np.isinf(train_data)))
# for cols in train_data.columns:
#     train_inf = np.any(np.isfinite(train_data[cols]))
#     test_inf = np.any(np.isfinite(test_data[cols]))
#     if train_inf == False:
#         print(train_inf)
#     if test_inf == False:
#         print(test_inf)

###########################筛选特征###################################
useless = [
      'QDUR_Y_min', 'rank_QEND_Y_max',
       'HY_12.0', 'rank_QBEGIN_Y_max', 'rank_QBEGIN_Y_mean', 'QDUR_Y_median',
       'QDUR_Y_max', 'rank_QEND_Y_min', 'HY_11.0', 'BRAENDRATIO', 'HY_7.0',
       'HY_8.0', 'HY_9.0', 'HY_6.0', 'HY_45.0', 'HY_16.0', 'HY_25.0', 'QDUR_Y',
       'ETYPE_2', 'ETYPE_1', 'HY_96.0', 'HY_94.0', 'HY_93.0', 'HY_91.0',
       'HY_90.0', 'QTYPE_2.0', 'HY_84.0', 'breakfaith_ENDY_min',
       'breakfaith_ENDY_max', 'breakfaith_ENDY_median', 'breakfaith_ENDY_mean',
       'breakfaith_ENDY_count', 'HY_76.0', 'QTYPE_3.0', 'QBEGIN_Y_count',
       'breakfaith_ENDM', 'breakfaith_ENDY', 'HY_60.0', 'HY_57.0', 'HY_56.0',
       'HY_55.0', 'HY_53.0', 'QEND_Y_mean', 'HY_28.0', 'HY_95.0'
]
# for feature in useless:
#     train_data.pop(feature)
#     test_data.pop(feature)


################################model######################################################
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

GB_param_grid = {
    # 'n_estimators' : [800,1500,1000], #
    # 'learning_rate' : [0.02,0.012], #
    'max_depth' : [18,19,20],       # 16-20 18 19
    'min_samples_split' : [2000,2250,2500], # 2000左右 2000 2250
    # 'min_samples_leaf' : [30,50,70], # 20左右 30 50
    # 'max_features' : [0.08,0.05,0.12]
    # 'subsample' : [0.6,0.7,0.75,0.85,0.9]
}

GBoost = GradientBoostingRegressor(max_features='sqrt',n_estimators=200, # 0.05，200
                                   max_depth= 18,learning_rate=0.05, # lr:0.01,n_esti:1000
                                   min_samples_leaf=30,min_samples_split=2000,
                                   loss='huber', random_state =5,
                                   subsample=0.8,
                                   )

# gsGBoost = GridSearchCV(GBoost,GB_param_grid,cv=3,scoring='roc_auc',verbose=5,return_train_score = True)
#
# gsGBoost.fit(train_data,y_train)
# GB_best = gsGBoost.best_estimator_
# print(GB_best)
# # Best score
# print(gsGBoost.best_score_)
# #
# auc = rmsle_cv(GBoost,train_data,y_train)
# print('Gboost AUC : ',auc)
#
GBoost.fit(train_data,y_train)
y_hat = GBoost.predict(test_data)
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
submission_data.to_csv(path_or_buf='../../output/gbdt.csv',index=None,header=True)