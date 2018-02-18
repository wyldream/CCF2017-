import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_summary import DataFrameSummary
import tables

data_path = "../../input/"

train_data = pd.read_csv(data_path+'initData/train.csv')
test_data = pd.read_csv(data_path+'initData/evaluation_public.csv')
entbase_data = pd.read_csv(data_path+'initData/1entbase.csv')

######## 读入、整合初赛数据

data_path1 = '../../../preliminary/public/'
train_data1 = pd.read_csv(data_path1+'train.csv')
entbase_data1 = pd.read_csv(data_path1 + '1entbase.csv')
entbase_data = pd.concat([entbase_data,entbase_data1])
train_data = pd.concat([train_data,train_data1])

##########

all_data = pd.concat([train_data,test_data])

all_data = pd.merge(all_data,entbase_data,how='left',on='EID') \
    .assign(rank_RGYEAR = lambda x: x.RGYEAR.rank()) \
    .assign(rank_ZCZB = lambda x: x.ZCZB.rank()) \
    .assign(rank_MPNUM = lambda x: x.MPNUM.rank()) \
    .assign(rank_INUM = lambda x: x.INUM.rank()) \
    .assign(rank_FINZB = lambda x: x.FINZB.rank()) \
    .assign(rank_FSTINUM = lambda x: x.FSTINUM.rank()) \
    .assign(rank_TZINUM = lambda x: x.TZINUM.rank()) \
    .assign(NUM = lambda x:x.ENUM+x.FSTINUM+x.INUM+x.MPNUM+x.TZINUM) \
    .assign(log_NUM = lambda x: np.log1p(x.NUM)).drop('NUM',1) \
    .assign(log_ZCZB = lambda x: np.log1p(x.ZCZB)).drop('ZCZB',1) \
    .assign(log_INUM = lambda x: np.log1p(x.INUM)).drop('INUM',1) \
    .assign(log_FSTINUM = lambda x : np.log1p(x.FSTINUM)).drop('FSTINUM',1) \
    .assign(log_MPNUM = lambda  x : np.log1p(x.MPNUM)).drop('MPNUM',1) \
    .assign(log_ENUM = lambda  x: np.log1p(x.ENUM)).drop('ENUM',1) \
    .assign(FINZB = lambda x:np.log1p(x.FINZB)) \
    .assign(PROV = lambda x:x.PROV.fillna(13)) \
    # .sort_values(['EID']) \

###############################组合特征##############################################
NUM = ['log_MPNUM','log_INUM','log_FSTINUM','log_ENUM']
tmp = all_data[['EID']]
for num1 in NUM:
    for num2 in NUM:
        if num1 != num2:
            num = all_data[num1] + all_data[num2]
            num = pd.DataFrame(num,index=[num.index])
            tmp = pd.concat([tmp,num],axis=1)
tmp.columns = ['EID','NUM1','NUM2','NUM3','NUM4','NUM5','NUM6','NUM7','NUM8','NUM9','NUM10','NUM11','NUM12']
for numi in ['NUM4','NUM7','NUM8','NUM10','NUM11','NUM12']:
    tmp = tmp.drop(numi,1) # 去重
all_data = pd.merge(all_data,tmp,'left','EID')

#####################################################################################

all_data = pd.get_dummies(all_data,columns=['PROV'],prefix='prov')
all_data = pd.get_dummies(all_data,columns=['HY'],prefix='HY')
all_data = pd.get_dummies(all_data,columns=['ETYPE'],prefix='ETYPE')
# all_data = pd.get_dummies(all_data,columns=['RGYEAR'],prefix='RGYEAR')

###############################过滤特征################################################
HY = [
    'HY_45.0', 'HY_7.0', 'HY_8.0', 'HY_57.0',
    'HY_6.0', 'HY_93.0', 'HY_94.0', 'HY_9.0', 'HY_28.0', 'HY_11.0',
    'HY_12.0', 'HY_53.0', 'HY_16.0', 'HY_91.0', 'HY_95.0', 'HY_90.0',
    'HY_76.0', 'HY_96.0'
      ]
for hy in HY:
    all_data.pop(hy)
#######################################################################################

print(all_data)
#
all_data.to_hdf(data_path + 'processedData/entbase.h5','w',complib='blosc',compleve=5)


