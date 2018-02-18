import numpy as np
import pandas as pd
import tables
import re
from UDFs import *

data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
all_data = pd.concat([train_data,test_data])

qualification_data = pd.read_csv(data_path+'initData/10qualification.csv') # 读取错误。将文件改为utf-8格式

def str_to_num(x):
    x = str(x)
    x = re.sub("\D","",x)
    if x == "":
        x = '0'
    x = float(x)
    return x

all_data = pd.merge(all_data,qualification_data,'left','EID') \
    .assign(QBEGIN_Y = lambda x:x.BEGINDATE.map(lambda x:x[:4] if x is not np.nan else x)) \
    .assign(QEND_Y = lambda x:x.EXPIRYDATE.map(lambda x:x[:4] if x is not np.nan else x)) \
    .assign(QBEGIN_Y = lambda x:x.QBEGIN_Y.astype(float)) \
    .assign(QEND_Y = lambda x:x.QEND_Y.astype(float)) \
    .assign(QDUR_Y = lambda x:x.QEND_Y - x.QBEGIN_Y) \
    .assign(rank_QBEGIN_Y = lambda x:x.QBEGIN_Y.rank()) \
    .assign(rank_QEND_Y = lambda x:x.QEND_Y.rank()) \
    # .assign(QBEGIN_M = lambda x:x.BEGINDATE.map(lambda x:x[5:6] if x is not np.nan else x)) \
    # .assign(QEND_M = lambda x:x.EXPIRYDATE.map(lambda x:x[5:6] if x is not np.nan else x)) \
    # .assign(RECDATE_Y = lambda x:x.RECDATE_Y.astype(float)) \
    # .assign(RECDATE_M = lambda x:x.RECDATE_M.astype(float)) \

BEGINY = gr_agg(all_data,'EID','QBEGIN_Y','count','mean','median','max','min')
ENDY = gr_agg(all_data,'EID','QEND_Y','mean','median','max','min')
DURY = gr_agg(all_data,'EID','QDUR_Y','mean','median','max','min')
RANKBEGINY = gr_agg(all_data,'EID','rank_QBEGIN_Y','mean','median','max','min')
RANKENDY = gr_agg(all_data,'EID','rank_QEND_Y','mean','median','max','min')

BEGINY.QBEGIN_Y_count = np.log1p(BEGINY.QBEGIN_Y_count)

all_data = pd.get_dummies(all_data,columns=['ADDTYPE'],prefix='QTYPE')

# all_data = all_data.groupby(all_data.EID,as_index = False).mean() # EID为什么没有了
all_data = all_data.groupby(all_data.EID).sum()
all_data = all_data.reset_index()

for data in [BEGINY,ENDY,DURY,RANKBEGINY,RANKENDY]:
    all_data = pd.merge(all_data,data,'left','EID')

print(all_data)

all_data.to_hdf(data_path + 'processedData/qualification.h5','w',complib='blosc',compleve=5)