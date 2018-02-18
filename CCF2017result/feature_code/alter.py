import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_summary import DataFrameSummary
import tables
import re
from UDFs import *


data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
alter_data = pd.read_csv(data_path+'initData/2alter.csv')

######## 读入、整合初赛数据

data_path1 = '../../../preliminary/public/'
train_data1 = pd.read_csv(data_path1+'train.csv')
alter_data1 = pd.read_csv(data_path1 + '2alter.csv')
alter_data = pd.concat([alter_data,alter_data1])
train_data = pd.concat([train_data,train_data1])

##########

all_data = pd.concat([train_data,test_data])


def str_to_num(x):
    x = str(x)
    x = re.sub("\D","",x)
    if x == "":
        x = '0'
    x = float(x)
    return x

all_data = pd.merge(all_data,alter_data,'left','EID') \
    .assign(ALT_Y = lambda x: x.ALTDATE.apply(lambda x: float(x[:4]) if x is not np.nan else x)) \
    .assign(ALT_M = lambda x: x.ALTDATE.apply(lambda x: float(x[-2:]) if x is not np.nan else x)) \
    .drop('ALTDATE',1) \
    .assign(ALTBE = lambda x: x.ALTBE.map(str_to_num))\
    .assign(ALTAF = lambda x: x.ALTAF.map(str_to_num)) \
    .assign(ALTDIFF = lambda x: x.ALTAF - x.ALTBE) \
    .assign(rank_ALTDIFF = lambda x:x.ALTDIFF.rank()) \
    .assign(rank_ALTAF = lambda x:x.ALTAF.rank()) \
    .assign(rank_ALTBE = lambda x:x.ALTBE.rank())
    # .assign(ALTBE = lambda x: np.log1p(np.log1p(x.ALTBE))) \
    # .assign(ALTAF = lambda x:np.log1p(np.log1p(x.ALTAF))) \
    # .assign(ALTDIFF = lambda x:np.log1p(x.ALTDIFF)) # 为什么为nan

# gr_agg为UDFs中定义的通用方法
ALT_Y = gr_agg(all_data, 'EID', 'ALT_Y','max', 'min','mean','median')
ALT_M = gr_agg(all_data, 'EID', 'ALT_M','count', 'max', 'min','mean','median')
ALTBE = gr_agg(all_data, 'EID', 'ALTBE', 'count','max', 'min','mean','median') # ,'std','skew'
ALTAF = gr_agg(all_data, 'EID', 'ALTAF', 'max', 'min','mean','median') # 'std','skew'
ALTDIFF = gr_agg(all_data, 'EID', 'ALTDIFF','max', 'min','mean','median') # ,'std','skew'
RANKAF = gr_agg(all_data, 'EID', 'rank_ALTAF','mean')
RANKBE = gr_agg(all_data, 'EID', 'rank_ALTBE','mean')
RANKDIFF = gr_agg(all_data, 'EID', 'rank_ALTDIFF','mean')


all_data = pd.get_dummies(all_data,columns=['ALTERNO'],prefix='ALTNO')
all_data = pd.get_dummies(all_data,columns=['ALT_Y'],prefix='ALT_Y')

# print(ALT_M.ALT_M_count)
# print(ALTBE.ALTBE_count)
# ALTBERATIO = (ALTBE.ALTBE_count/ALT_M.ALT_M_count)
# print(ALTBERATIO)

all_data = all_data.groupby(all_data.EID).sum()
all_data = all_data.reset_index()
print(all_data)

ALT_M.ALT_M_count = np.log1p(ALT_M.ALT_M_count)
ALTBE.ALTBE_count = np.log1p(ALTBE.ALTBE_count)

ALTBE.ALTBE_max = np.log1p(ALTBE.ALTBE_max)
ALTBE.ALTBE_min = np.log1p(ALTBE.ALTBE_min)
ALTBE.ALTBE_mean = np.log1p(ALTBE.ALTBE_mean)
ALTBE.ALTBE_median = np.log1p(ALTBE.ALTBE_median)
all_data.ALTBE = np.log1p(all_data.ALTBE)

ALTAF.ALTAF_max = np.log1p(ALTAF.ALTAF_max)
ALTAF.ALTAF_min = np.log1p(ALTAF.ALTAF_min)
ALTAF.ALTAF_mean = np.log1p(ALTAF.ALTAF_mean)
ALTAF.ALTAF_median = np.log1p(ALTAF.ALTAF_median)
all_data.ALTAF = np.log1p(all_data.ALTAF)

ALTDIFF.ALTDIFF_max = np.log1p(ALTDIFF.ALTDIFF_max)
ALTDIFF.ALTDIFF_min = np.log1p(ALTDIFF.ALTDIFF_min)
ALTDIFF.ALTDIFF_mean = np.log1p(ALTDIFF.ALTDIFF_mean)
ALTDIFF.ALTDIFF_median = np.log1p(ALTDIFF.ALTDIFF_median)
all_data.ALTDIFF = np.log1p(all_data.ALTDIFF)

####################################组合特征###################################################
# ALTNO = ['ALTNO_99','ALTNO_03','ALTNO_12','ALTNO_02','ALTNO_A_015']
###############################################################################################

for data in [ALT_Y,ALT_M,ALTBE,ALTAF,RANKAF,RANKBE,RANKDIFF]:
    all_data = pd.merge(all_data,data,'left','EID')

print(all_data)

all_data.to_hdf(data_path + 'processedData/alter.h5','w',complib='blosc',compleve=5)
