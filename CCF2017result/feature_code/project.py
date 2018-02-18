import numpy as np
import pandas as pd
import tables
import re
from UDFs import *

data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
project_data = pd.read_csv(data_path+'initData/6project.csv')

######## 读入、整合初赛数据

data_path1 = '../../../preliminary/public/'
train_data1 = pd.read_csv(data_path1+'train.csv')
project_data1 = pd.read_csv(data_path1 + '6project.csv')
project_data = pd.concat([project_data,project_data1])
train_data = pd.concat([train_data,train_data1])
# print(project_data1)
# print(project_data)
# print(train_data)

##########


all_data = pd.concat([train_data,test_data])

all_data = pd.merge(all_data,project_data,'left','EID') \
    .assign(DJDATE_Y = lambda x:x.DJDATE.map(lambda x:x[:4] if x is not np.nan else x)) \
    .assign(DJDATE_M = lambda x:x.DJDATE.map(lambda x:x[5:] if x is not np.nan else x)) \
    .assign(DJDATE_Y = lambda x:x.DJDATE_Y.astype(float)) \
    .assign(DJDATE_M = lambda x:x.DJDATE_M.astype(float)) \
    .assign(rank_DJDATE_Y = lambda x:x.DJDATE_Y.rank()) \
    .drop('DJDATE',1) \
    .drop('TYPECODE',1)

DJDATEY = gr_agg(all_data,'EID','DJDATE_Y','mean','median','max','min')
DJDATEM = gr_agg(all_data,'EID','DJDATE_M','count','mean','median','max','min')
RANKY = gr_agg(all_data,'EID','rank_DJDATE_Y','mean','median','max','min')
# TYPECODE = gr_agg(all_data,'EID','TYPECODE','mean','median','max','min')

all_data = pd.get_dummies(all_data,columns=['DJDATE_Y'],prefix='DJDATEY')

DJDATEM.DJDATE_M_count = np.log1p(DJDATEM.DJDATE_M_count)
# print(DJDATEY.DJDATE_Y_count.skew())

all_data = all_data.groupby(all_data.EID,as_index = False).sum()

for data in [DJDATEY,DJDATEM,RANKY]:
    all_data = pd.merge(all_data,data,'left','EID')

# all_data = all_data[['EID','ENDDATE','TARGET','DJDATE_M_count']]

print(all_data)

all_data.to_hdf(data_path + 'processedData/project.h5','w',complib='blosc',compleve=5)