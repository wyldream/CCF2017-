import numpy as np
import pandas as pd
import tables
import re
from UDFs import *

data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
recruit_data = pd.read_csv(data_path+'initData/9recruit.csv')

######## 读入、整合初赛数据

data_path1 = '../../../preliminary/public/'
train_data1 = pd.read_csv(data_path1+'train.csv')
recruit_data1 = pd.read_csv(data_path1 + '9recruit.csv')
recruit_data1.rename(columns = {'RECRNUM':'PNUM'},inplace=True)
recruit_data = pd.concat([recruit_data,recruit_data1])
train_data = pd.concat([train_data,train_data1])
# print(recruit_data1)
# print(recruit_data)
# print(train_data)

##########

all_data = pd.concat([train_data,test_data])

def str_to_num(x):
    if x == '若干': # 中值填充若干（初赛数据中的中值）
        x = 9
    x = str(x)
    x = re.sub("\D","",x)
    if x == "":
        x = '9' # 中值填充空
    x = float(x)
    return x

# recruit_data.PNUM = recruit_data.PNUM.map(str_to_num)
# print(recruit_data.PNUM.describe())

all_data = pd.merge(all_data,recruit_data,'left','EID') \
    .assign(RECDATE_Y = lambda x:x.RECDATE.map(lambda x:x[:4] if x is not np.nan else x)) \
    .assign(RECDATE_M = lambda x:x.RECDATE.map(lambda x:x[5:] if x is not np.nan else x)) \
    .assign(RECDATE_Y = lambda x:x.RECDATE_Y.astype(float)) \
    .assign(RECDATE_M = lambda x:x.RECDATE_M.astype(float)) \
    .assign(PNUM = lambda x:x.PNUM.map(str_to_num)) \
    .assign(rank_PNUM = lambda x:x.PNUM.rank()) \
    .assign(rank_RECY = lambda x:x.RECDATE_Y.rank()) \
    .drop('POSCODE',1) \
    .drop('RECDATE',1)

# print(all_data.rank_RECY)
# print(all_data.rank_RECY.describe())
# print(all_data.rank_RECY.value_counts())

# RECDATEY = gr_agg(all_data,'EID','RECDATE_Y','mean','median','max','min')
PNUM = gr_agg(all_data,'EID','PNUM','count','mean','median','max','min')
RANKPNUM = gr_agg(all_data,'EID','rank_PNUM','mean','median','max','min')
# RANKRECY = gr_agg(all_data,'EID','rank_RECY','mean','median','max','min')

all_data = pd.get_dummies(all_data,columns=['WZCODE'],prefix='WZCODE')
all_data = pd.get_dummies(all_data,columns=['RECDATE_Y'],prefix='RECY')
all_data = pd.get_dummies(all_data,columns=['rank_RECY'],prefix='RANK_RECY')

all_data = all_data.groupby(all_data.EID).sum()
all_data = all_data.reset_index()

PNUM.PNUM_count = np.log1p(PNUM.PNUM_count)
PNUM.PNUM_mean = np.log1p(PNUM.PNUM_mean)
PNUM.PNUM_median = np.log1p(PNUM.PNUM_median)
PNUM.PNUM_max = np.log1p(PNUM.PNUM_max)
PNUM.PNUM_min = np.log1p(PNUM.PNUM_min)
all_data.PNUM = np.log1p(all_data.PNUM)

for data in [PNUM,RANKPNUM]:
    all_data = pd.merge(all_data,data,'left','EID')

print(all_data)

all_data.to_hdf(data_path + 'processedData/recruit.h5','w',complib='blosc',compleve=5)
