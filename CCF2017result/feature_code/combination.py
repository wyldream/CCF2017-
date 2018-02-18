import numpy as np
import pandas as pd
import tables

data_path = "../../input/"

all_data = pd.read_hdf(data_path+'processedData/all_data.h5')
# all_data = pd.read_csv(data_path+'processedData/all_data.csv')

def combination(data,features):
    i = 0
    j = 0
    comb = data[['EID']]
    while i < features.__len__():
        col1 = features[i]
        j = i+1
        while j < features.__len__():
            col2 = features[j]
            tmp = data[col1] + data[col2]
            tmp = pd.DataFrame(tmp,index=[tmp.index])
            comb = pd.concat([comb,tmp],axis = 1)
            j = j + 1
        i = i + 1
    return comb
########################################积极组合特征#############################################################
poscorr = ['B_REYEAR_count','BTYEAR_count','right_ASKY_count',
           'DJDATE_M_count','PNUM_count','PNUM_mean']

# poscorr = ['log_ZCZB','BTBL','ALT_M_count','B_REYEAR_count','BTYEAR_count',
#            'right_ASKY_count','DJDATE_M_count','PNUM_count','PNUM_mean']

comb = combination(all_data,poscorr)

comb.columns = ['EID','comb1','comb2','comb3','comb4','comb5',
                'comb6','comb7','comb8','comb9','comb10',
                'comb11','comb12','comb13','comb14','comb15',
                ]

all_data = pd.merge(all_data,comb,'left','EID')
#########################################消极组合特征####################################################################
negcorr = ['LAWDATE_Y_count','LAWAMOUNT_mean','breakfaith_FBY_count']
comb1 = combination(all_data,negcorr)
comb1.columns = ['EID','neg1','neg2','neg3']
all_data = pd.merge(all_data,comb1,'left','EID')
print(all_data)
###############################################################################################################

all_data.to_hdf(data_path + 'processedData/all_data.h5','w',complib='blosc',compleve=5)
# all_data.to_csv(data_path + 'processedData/all_data.csv',index=None)