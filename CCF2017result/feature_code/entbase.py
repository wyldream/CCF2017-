import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_summary import DataFrameSummary
import tables
import copy

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
    # .assign(ETYPE_HY_count = all_data.groupby(['ETYPE','HY'])['EID'].count())
    # .sort_values(['EID']) \
#############################今日最佳的特征##########################################################
# 同行业类型公司数
HY_count = all_data.groupby('HY')['HY'].count()
HY_count = pd.DataFrame(HY_count.values,columns=['HYCOUNT'],index=[HY_count.index])
all_data = pd.merge(all_data,HY_count,'left',left_on='HY',right_index=True)

# 同企业类型公司数
ETYPE_count = all_data.groupby('ETYPE')['ETYPE'].count()
ETYPE_count = pd.DataFrame(ETYPE_count.values,columns=['ETYPECOUNT'],index=[ETYPE_count.index])
all_data = pd.merge(all_data,ETYPE_count,'left',left_on='ETYPE',right_index=True)

# 同时间成立的公司数
RGYEAR_count = all_data.groupby('RGYEAR')['RGYEAR'].count()
RGYEAR_count = pd.DataFrame(RGYEAR_count.values,columns=['RGYEARCOUNT'],index=[RGYEAR_count.index])
all_data = pd.merge(all_data,RGYEAR_count,'left',left_on='RGYEAR',right_index=True)

# 限定HY，按照注册资本金额排序
HY_ZCZB_rank = all_data.groupby('HY')['log_ZCZB'].rank()
HY_ZCZB_rank = pd.DataFrame(HY_ZCZB_rank.values,columns=['HYZCZBRANK'],index=[HY_ZCZB_rank.index])
all_data = pd.concat([all_data,HY_ZCZB_rank],axis=1)

# 限定ETYPE，按照注册资本排序
ETYPE_ZCZB_rank = all_data.groupby('ETYPE')['log_ZCZB'].rank()
ETYPE_ZCZB_rank = pd.DataFrame(ETYPE_ZCZB_rank.values,columns=['ETYPEZCZBRANK'],index=[ETYPE_ZCZB_rank.index])
all_data = pd.concat([all_data,ETYPE_ZCZB_rank],axis=1)

# 限定RGYEAR，按照注册资本排序
RGYEAR_ZCZB_rank = all_data.groupby('RGYEAR')['log_ZCZB'].rank()
RGYEAR_ZCZB_rank = pd.DataFrame(RGYEAR_ZCZB_rank.values,columns=['RGYEARZCZBRANK'],index=[RGYEAR_ZCZB_rank.index])
all_data = pd.concat([all_data,RGYEAR_ZCZB_rank],axis=1)

# 同企业类型同行业类型数量
ETYPE_HY_count = all_data.groupby(['ETYPE','HY'],as_index = False)['EID'].count() # 以两列聚合、合并
ETYPE_HY_count.rename(columns = {'EID':'ETYPEHYCOUNT'},inplace = True)
all_data = pd.merge(all_data,ETYPE_HY_count,'left',on=['ETYPE','HY'])

# 同HY同RGYEAR
RGYEAR_HY_count = all_data.groupby(['RGYEAR','HY'],as_index = False)['EID'].count() # 以两列聚合、合并
RGYEAR_HY_count.rename(columns = {'EID':'RGYEARHYCOUNT'},inplace = True)
all_data = pd.merge(all_data,RGYEAR_HY_count,'left',on=['RGYEAR','HY'])

# 同ETYPE同RGYEAR
RGYEAR_ETYPE_count = all_data.groupby(['RGYEAR','ETYPE'],as_index = False)['EID'].count() # 以两列聚合、合并
RGYEAR_ETYPE_count.rename(columns = {'EID':'RGYEARETYPECOUNT'},inplace = True)
all_data = pd.merge(all_data,RGYEAR_ETYPE_count,'left',on=['RGYEAR','ETYPE'])
#######################################今日最佳特征第二弹##########################################
# 2017-RGYEAR
# all_data = all_data.assign(DIFFRGYEAR = lambda x:x.RGYEAR.map(lambda x:2017-x))

# 对排序后的FINZB等量划分为8份
# all_data = all_data.assign(cate_rank_FINZB = copy.copy(all_data.rank_FINZB))
# all_data.cate_rank_FINZB.loc[(all_data.rank_FINZB > 46) & (all_data.rank_FINZB <= 5540.5) ]         = 1
# all_data.cate_rank_FINZB.loc[(all_data.rank_FINZB > 5540.5) & (all_data.rank_FINZB <= 5540.5) ]     = 2
# all_data.cate_rank_FINZB.loc[(all_data.rank_FINZB > 11502) & (all_data.rank_FINZB <= 16379) ]       = 3
# all_data.cate_rank_FINZB.loc[(all_data.rank_FINZB > 16379) & (all_data.rank_FINZB <= 24193.5) ]     = 4
# all_data.cate_rank_FINZB.loc[(all_data.rank_FINZB > 24193.5) & (all_data.rank_FINZB <= 29533.375) ] = 5
# all_data.cate_rank_FINZB.loc[(all_data.rank_FINZB > 29533.375) & (all_data.rank_FINZB <= 34214.5) ] = 6
# all_data.cate_rank_FINZB.loc[(all_data.rank_FINZB > 34214.5) & (all_data.rank_FINZB <= 41497.5) ]   = 7
# all_data.cate_rank_FINZB.loc[(all_data.rank_FINZB > 41497.5) & (all_data.rank_FINZB <= 47246.5) ]   = 8
#
# all_data = all_data.assign(cate_rank_ZCZB = copy.copy(all_data.rank_ZCZB))
# all_data.cate_rank_ZCZB.loc[(all_data.rank_ZCZB > 51) & (all_data.rank_ZCZB <= 82190) ]              = 1
# all_data.cate_rank_ZCZB.loc[(all_data.rank_ZCZB > 82190) & (all_data.rank_ZCZB <= 204831) ]          = 2
# all_data.cate_rank_ZCZB.loc[(all_data.rank_ZCZB > 204831) & (all_data.rank_ZCZB <= 324904.5) ]       = 3
# all_data.cate_rank_ZCZB.loc[(all_data.rank_ZCZB > 324904.5) & (all_data.rank_ZCZB <= 393021.0) ]     = 4
# all_data.cate_rank_ZCZB.loc[(all_data.rank_ZCZB > 393021.0) & (all_data.rank_ZCZB <= 491139.5) ]     = 5
# all_data.cate_rank_ZCZB.loc[(all_data.rank_ZCZB > 491139.5) & (all_data.rank_ZCZB<= 589507.0) ]     = 6
# all_data = pd.get_dummies(all_data,columns=['cate_rank_ZCZB'],prefix='CATEZCZB')
#
# print(all_data.rank_INUM.describe())
# divide = pd.qcut(all_data.rank_INUM,5)
########################################组合特征####################################################

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

all_data.to_hdf(data_path + 'processedData/entbase.h5','w',complib='blosc',compleve=5)


