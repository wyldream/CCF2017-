import numpy as np
import pandas as pd
import tables
import re
from UDFs import *

data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
invest_data = pd.read_csv(data_path+'initData/4invest.csv')

######## 读入、整合初赛数据

data_path1 = '../../../preliminary/public/'
train_data1 = pd.read_csv(data_path1+'train.csv')
invest_data1 = pd.read_csv(data_path1 + '4invest.csv')
invest_data = pd.concat([invest_data,invest_data1])
train_data = pd.concat([train_data,train_data1])

##########

all_data = pd.concat([train_data,test_data])

all_data = pd.merge(all_data,invest_data,'left','EID') \
    .assign(BTEID = lambda x:x.BTEID.fillna(0)) \
    .assign(char_BTEID = lambda x:x.BTEID.map(lambda x:x[:1] if type(x) == str else 't')) \
    .assign(num_BTEID = lambda x:x.BTEID.map(lambda x:x[1:] if type(x) == str else x)) \
    .assign(num_BTEID = lambda x:x.num_BTEID.astype(float)) \
    .assign(BTDUR = lambda x:x.BTENDYEAR - x.BTYEAR) \
    .assign(rank_BTYEAR = lambda x:x.BTYEAR.rank()) \
    .assign(rank_BTBL = lambda x:x.BTBL.rank()) \
    .drop('BTEID',1) \

BTYEAR = gr_agg(all_data,'EID','BTYEAR','count','mean','median','max','min','sum')
BTENDYEAR = gr_agg(all_data,'EID','BTENDYEAR','count','mean','median','max','min')
BTDUR = gr_agg(all_data,'EID','BTDUR','mean','median','max','min')
BTBL = gr_agg(all_data,'EID','BTBL','mean','median','max','min')
num_BTEID = gr_agg(all_data,'EID','num_BTEID','mean','median','max','min')
RANKBTYEAR = gr_agg(all_data,'EID','rank_BTYEAR','mean','median','max','min')
RANKBTBL = gr_agg(all_data,'EID','rank_BTBL','mean','median','max','min')

BTYEAR.BTYEAR_count = np.log1p(BTYEAR.BTYEAR_count)
BTENDYEAR.BTENDYEAR_count = np.log1p(BTENDYEAR.BTENDYEAR_count)

#########################################leak###########################################################
# train_data = all_data[all_data.TARGET.notnull()]
# test_data = all_data[all_data.TARGET.isnull()]
# test_data.drop('BTENDYEAR',1,inplace = True)
# leak_data = pd.merge(all_data[['BTEID','BTENDYEAR']],test_data,'inner',left_on='BTEID',right_on='EID')
# print(leak_data)
# # print(leak_data.BTENDYEAR.describe())
# leak_data = leak_data[leak_data.BTENDYEAR.notnull()] # 已经关闭的才是leak
# print(leak_data)

######################################################################################################

all_data = pd.get_dummies(all_data,columns=['char_BTEID'],prefix='BTEID')
# all_data = pd.get_dummies(all_data,columns=['BTBL_grade'],prefix='BTBLGRADE') # 根据持股比例分级

all_data = all_data.groupby(all_data.EID).sum()
all_data = all_data.reset_index()

for data in [BTYEAR,BTENDYEAR,BTBL,num_BTEID,BTDUR,RANKBTBL,RANKBTYEAR]:
    all_data = pd.merge(all_data,data,'left','EID')

print(all_data)

all_data.to_hdf(data_path + 'processedData/invest.h5','w',complib='blosc',compleve=5)