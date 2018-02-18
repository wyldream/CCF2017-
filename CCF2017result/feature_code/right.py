import numpy as np
import pandas as pd
import tables
import re
from UDFs import *

data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
right_data = pd.read_csv(data_path + 'initData/5right.csv')

######## 读入、整合初赛数据

data_path1 = '../../../preliminary/public/'
train_data1 = pd.read_csv(data_path1+'train.csv')
right_data1 = pd.read_csv(data_path1 + '5right.csv')
right_data = pd.concat([right_data,right_data1])
train_data = pd.concat([train_data,train_data1])
print(right_data1)
print(right_data)
print(train_data)

##########

all_data = pd.concat([train_data,test_data])


def str_to_num(x):
    x = str(x)
    x = re.sub("\D","",x)
    if x == "":
        x = '0'
    x = float(x)
    return x

all_data = pd.merge(all_data,right_data,'left','EID') \
    .assign(right_ASKY = lambda x:x.ASKDATE.map(lambda x:x[:4] if x is not np.nan else x)) \
    .assign(right_ASKM = lambda x:x.ASKDATE.map(lambda x:x[5:] if x is not np.nan else x)) \
    .assign(right_FBY = lambda x:x.FBDATE.map(lambda x:x[:4] if x is not np.nan else x)) \
    .assign(right_FBM = lambda x:x.FBDATE.map(lambda x:x[5:] if x is not np.nan else x)) \
    .assign(num_rightcode = lambda x:x.TYPECODE.map(str_to_num)) \
    .assign(num_rightcode = lambda x:x.num_rightcode.astype(float)) \
    .assign(right_ASKY = lambda x:x.right_ASKY.astype(float)) \
    .assign(right_ASKM = lambda x:x.right_ASKM.astype(float)) \
    .assign(right_FBY = lambda x:x.right_FBY.astype(float)) \
    .assign(right_FBM = lambda x:x.right_FBM.astype(float)) \
    .assign(rank_right_ASKY = lambda x:x.right_ASKY.rank()) \
    .assign(rank_right_FBY = lambda x:x.right_FBY.rank()) \
    .drop('ASKDATE',1) \
    .drop('FBDATE',1) \
    .drop('TYPECODE',1)

ASKY = gr_agg(all_data,'EID','right_ASKY','count','mean','median','max','min')
FBY = gr_agg(all_data,'EID','right_FBY','count','mean','median','max','min')
RIGHTCODE = gr_agg(all_data,'EID','num_rightcode','mean','median','max','min')
RANKASKY = gr_agg(all_data,'EID','rank_right_ASKY','mean','median','max','min')
RANKFBY = gr_agg(all_data,'EID','rank_right_FBY','mean','median','max','min')
# ASKM = gr_agg(all_data,'EID','right_ASKM','mean','median','max','min')
# FBM = gr_agg(all_data,'EID','right_FBM','mean','median','max','min')

ASKY.right_ASKY_count = np.log1p(ASKY.right_ASKY_count)
FBY.right_FBY_count = np.log1p(FBY.right_FBY_count)
# print(FBY.right_FBY_count.skew())

all_data = pd.get_dummies(all_data,columns = ['RIGHTTYPE'],prefix = 'RIGHTTYPE')
all_data = all_data.groupby(all_data.EID,as_index=False).sum()

for data in [ASKY,FBY,RIGHTCODE,RANKASKY,RANKFBY]:
    all_data = pd.merge(all_data,data,'left','EID')

print(all_data)

all_data.to_hdf(data_path + 'processedData/right.h5','w',complib='blosc',compleve=5)