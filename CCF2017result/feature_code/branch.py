import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tables
import re
from UDFs import *

data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
branch_data = pd.read_csv(data_path+'initData/3branch.csv')

######## 读入、整合初赛数据

data_path1 = '../../../preliminary/public/'
train_data1 = pd.read_csv(data_path1+'train.csv')
branch_data1 = pd.read_csv(data_path1 + '3branch.csv')
branch_data = pd.concat([branch_data,branch_data1])
train_data = pd.concat([train_data,train_data1])

##########

def str_to_num(x):
    x = str(x)
    x = re.sub("\D","",x)
    if x == "":
        x = '0'
    x = float(x)
    return x

all_data = pd.concat([train_data,test_data])

all_data = pd.merge(all_data,branch_data,'left','EID') \
    .assign(B_DURYEAR = lambda x:x.B_ENDYEAR-x.B_REYEAR) \
    .assign(num_TYPECODE = lambda x:x.TYPECODE.apply(str_to_num)) \
    .assign(rank_B_REYEAR = lambda x:x.B_REYEAR.rank()) \
    .assign(rank_B_ENDYEAR = lambda x:x.B_ENDYEAR.rank()) \
    .assign(rank_B_DURYEAR = lambda x:x.B_DURYEAR.rank()) \
    .drop('TYPECODE',1) \

REYEAR = gr_agg(all_data,'EID','B_REYEAR','count','mean','median','max','min','sum')
ENDYEAR = gr_agg(all_data,'EID','B_ENDYEAR','count','mean','median','max','min')
DURYEAR = gr_agg(all_data,'EID','B_DURYEAR','mean','median','max','min')
TYPECODE = gr_agg(all_data,'EID','num_TYPECODE','mean','median','max','min')
RANKREYEAR = gr_agg(all_data,'EID','rank_B_REYEAR','mean')
RANKENDYEAR = gr_agg(all_data,'EID','rank_B_ENDYEAR','mean')
RANKDURYEAR = gr_agg(all_data,'EID','rank_B_DURYEAR','mean')

########################################leak data#######################################################
# branch = all_data[['TYPECODE','B_ENDYEAR']]
# test_data = all_data[all_data.TARGET.isnull()]
# test_data.drop('B_ENDYEAR',1,inplace = True)
# test_data.drop('TYPECODE',1,inplace = True)
# branch = branch.assign(TYPECODE = branch.TYPECODE.map(lambda x:x.replace('br','') if type(x) == str else x))
# leak = pd.merge(branch,test_data,'inner',left_on='TYPECODE',right_on='EID')
# leak = leak[leak.B_ENDYEAR.notnull()]
# print(len(leak.EID.unique()))
# leak = leak.groupby(leak.EID,as_index = False).sum()
# leak_TARGET = leak[['EID','TARGET']]
# leak_TARGET.TARGET = 1
# print(leak_TARGET)
# leak_TARGET.to_hdf(data_path + 'processedData/leak.h5','w',complib='blosc',compleve=5)
# print('complete')

# pd.merge
######################################################################################################

BRAENDRATIO = (ENDYEAR.B_ENDYEAR_count/REYEAR.B_REYEAR_count).reset_index()
BRAENDRATIO = pd.DataFrame(BRAENDRATIO,columns=['BRAENDRATIO'],index=[REYEAR.EID]).reset_index()

all_data = all_data.groupby(all_data.EID,as_index=False).sum()

REYEAR.B_REYEAR_count = np.log1p(REYEAR.B_REYEAR_count)
ENDYEAR.B_ENDYEAR_count = np.log1p(ENDYEAR.B_ENDYEAR_count)

for data in [REYEAR,ENDYEAR,DURYEAR,TYPECODE,BRAENDRATIO]:
    all_data = pd.merge(all_data,data,'left','EID')

print(all_data)

all_data.to_hdf(data_path + 'processedData/branch.h5','w',complib='blosc',compleve=5)