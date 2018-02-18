import numpy as np
import pandas as pd
import tables
import re
from UDFs import *

data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
lawsuit_data = pd.read_csv(data_path+'initData/7lawsuit.csv')

######## 读入、整合初赛数据

data_path1 = '../../../preliminary/public/'
train_data1 = pd.read_csv(data_path1+'train.csv')
lawsuit_data1 = pd.read_csv(data_path1 + '7lawsuit.csv')
lawsuit_data = pd.concat([lawsuit_data,lawsuit_data1])
train_data = pd.concat([train_data,train_data1])
# print(lawsuit_data1)
# print(lawsuit_data)
# print(train_data)

##########

all_data = pd.concat([train_data,test_data])

all_data = pd.merge(all_data,lawsuit_data,'left','EID') \
    .assign(LAWDATE_Y = lambda x:x.LAWDATE.map(lambda x:x[:4] if x is not np.nan else x)) \
    .assign(LAWDATE_M = lambda x:x.LAWDATE.map(lambda x:x[5:7] if x is not np.nan else x)) \
    .assign(LAWDATE_Y = lambda x:x.LAWDATE_Y.astype(float)) \
    .assign(LAWDATE_M = lambda x:x.LAWDATE_M.astype(float)) \
    .assign(rank_LAWAMOUNT = lambda x:x.LAWAMOUNT.rank()) \
    .assign(rank_LAWADATE_Y = lambda x:x.LAWDATE_Y.rank()) \
    .drop('LAWDATE',1) \
    .drop('TYPECODE',1) \
    # .assign(log_LAWAMOUNT = lambda x:np.log1p(x.LAWAMOUNT)) \

LAWY = gr_agg(all_data,'EID','LAWDATE_Y','count','mean','median','max','min')
LAWAMOUNT = gr_agg(all_data,'EID','LAWAMOUNT','mean','median','max','min')
RANKLAWAMOUNT = gr_agg(all_data,'EID','rank_LAWAMOUNT','mean','median','max','min')
RANKLAWAY = gr_agg(all_data,'EID','rank_LAWADATE_Y','mean','median','max','min')
# TYPECODE = gr_agg(all_data,'EID','TYPECODE','mean','median','max','min')

all_data = pd.get_dummies(all_data,columns=['LAWDATE_Y'],prefix='LAWY')

all_data = all_data.groupby(all_data.EID,as_index=False).sum()

LAWY.LAWDATE_Y_count = np.log1p(LAWY.LAWDATE_Y_count)
LAWAMOUNT.LAWAMOUNT_mean = np.log1p(LAWAMOUNT.LAWAMOUNT_mean)
LAWAMOUNT.LAWAMOUNT_median = np.log1p(LAWAMOUNT.LAWAMOUNT_median)
LAWAMOUNT.LAWAMOUNT_max = np.log1p(LAWAMOUNT.LAWAMOUNT_max)
LAWAMOUNT.LAWAMOUNT_min = np.log1p(LAWAMOUNT.LAWAMOUNT_min)
all_data.LAWAMOUNT = np.log1p(all_data.LAWAMOUNT)
# print(all_data.LAWAMOUNT.skew())

for data in [LAWY,LAWAMOUNT,RANKLAWAMOUNT,RANKLAWAY]:
    all_data = pd.merge(all_data,data,'left','EID')

# print(all_data.info())
# all_data = all_data[['LAWDATE_Y_count','LAWAMOUNT_mean','ENDDATE','TARGET','EID']]
print(all_data)

all_data.to_hdf(data_path + 'processedData/lawsuit.h5','w',complib='blosc',compleve=5)