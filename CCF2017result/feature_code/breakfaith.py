import numpy as np
import pandas as pd
import tables
from UDFs import *

data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
breakfaith_data = pd.read_csv(data_path+'initData/8breakfaith.csv')

######## 读入、整合初赛数据

data_path1 = '../../../preliminary/public/'
train_data1 = pd.read_csv(data_path1+'train.csv')
breakfaith_data1 = pd.read_csv(data_path1 + '8breakfaith.csv')
breakfaith_data = pd.concat([breakfaith_data,breakfaith_data1])
train_data = pd.concat([train_data,train_data1])
# print(breakfaith_data1)
# print(breakfaith_data)
# print(train_data)

##########

all_data = pd.concat([train_data,test_data])

all_data = pd.merge(all_data,breakfaith_data,'left','EID') \
    .assign(breakfaith_FBY = lambda x:x.FBDATE.map(lambda x:x[:4] if x is not np.nan else x)) \
    .assign(breakfaith_ENDY = lambda x:x.SXENDDATE.map(lambda x:x[:4] if x is not np.nan else x)) \
    .assign(breakfaith_FBY = lambda x:x.breakfaith_FBY.astype(float)) \
    .assign(breakfaith_ENDY = lambda x:x.breakfaith_ENDY.astype(float)) \
    .assign(rank_breakfaith_FBY = lambda x:x.breakfaith_FBY.rank()) \
    .drop('FBDATE',1) \
    .drop('SXENDDATE',1) \
    .drop('TYPECODE',1) \
    # .assign(breakfaith_FBM = lambda x:x.FBDATE.map(lambda x:x[5:7] if x is not np.nan else x)) \
    # .assign(breakfaith_ENDM = lambda x:x.SXENDDATE.map(lambda x:x[5:] if x is not np.nan else x)) \
    # .assign(breakfaith_FBM = lambda x:x.breakfaith_FBM.astype(float)) \
    # .assign(breakfaith_ENDM = lambda x:x.breakfaith_ENDM.astype(float)) \

FBY = gr_agg(all_data,'EID','breakfaith_FBY','count','mean','median','max','min')
ENDY = gr_agg(all_data,'EID','breakfaith_ENDY','count','mean','median','max','min')
RANKFBY = gr_agg(all_data,'EID','rank_breakfaith_FBY','mean','median','max','min')

FBY.breakfaith_FBY_count = np.log1p(FBY.breakfaith_FBY_count)
ENDY.breakfaith_ENDY_count = np.log1p(ENDY.breakfaith_ENDY_count)
# print(FBY.breakfaith_FBY_count.skew())
# print(ENDY.breakfaith_ENDY_count.skew())

# print(all_data.breakfaith_FBY.value_counts())
all_data = pd.get_dummies(all_data,columns = ['breakfaith_FBY'],prefix = 'breakfaith_FBY')

all_data = all_data.groupby(all_data.EID,as_index=False).sum()

for data in [FBY,ENDY,RANKFBY]:
    all_data = pd.merge(all_data,data,'left','EID')

print(all_data)

all_data.to_hdf(data_path + 'processedData/breakfaith.h5','w',complib='blosc',compleve=5)