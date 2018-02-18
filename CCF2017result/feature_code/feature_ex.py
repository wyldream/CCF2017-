# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:40:51 2017

@author: Administrator
"""

import pandas as pd
path='C:/Luoshichao/ccf_3/model_data_2/'
df1=pd.read_csv(path+'df1_temp.csv')
df2=pd.read_csv(path+'df2_temp.csv')
df3=pd.read_csv(path+'df3_temp.csv')
df4=pd.read_csv(path+'df4_temp.csv')
df5=pd.read_csv(path+'df5_temp.csv')
df6=pd.read_csv(path+'df6_temp.csv')
df7=pd.read_csv(path+'df7_temp.csv')
df8=pd.read_csv(path+'df8_temp.csv')
df9=pd.read_csv(path+'df9_temp.csv')
df10=pd.read_csv(path+'df10_temp.csv')

data_ex=df1
for data in [df2,df3,df4,df5,df6,df7,df8,df9,df10]:
    data_ex=pd.merge(data_ex,data,on='EID',how='left')
    
#df2.columns    
data_ex['ALT_Y_max_RG']=data_ex['ALT_Y_max']-data_ex['RGYEAR']
data_ex['ALT_Y_min_RG']=data_ex['ALT_Y_min']-data_ex['RGYEAR']
data_ex.drop(['ALT_Y_max', 'ALT_Y_min'],axis=1,inplace=True)


#df3.columns 
data_ex['B_REYEAR_max_RG']=data_ex['B_REYEAR_max']-data_ex['RGYEAR']
data_ex['B_REYEAR_min_RG']=data_ex['B_REYEAR_min']-data_ex['RGYEAR']
data_ex.drop(['B_REYEAR_max', 'B_REYEAR_min'],axis=1,inplace=True)


#df4.columns 
data_ex['BTYEAR_max_RG']=data_ex['BTYEAR_max']-data_ex['RGYEAR']
data_ex['BTYEAR_min_RG']=data_ex['BTYEAR_min']-data_ex['RGYEAR']
data_ex.drop(['BTYEAR_max', 'BTYEAR_min'],axis=1,inplace=True)

#df5.columns 
data_ex['right_ASKY_max_RG']=data_ex['right_ASKY_max']-data_ex['RGYEAR']
data_ex['right_ASKY_min_RG']=data_ex['right_ASKY_min']-data_ex['RGYEAR']
data_ex.drop(['right_ASKY_max', 'right_ASKY_min'],axis=1,inplace=True)

#df6.columns
data_ex['DJDATE_Y_max_RG']=data_ex['DJDATE_Y_max']-data_ex['RGYEAR']
data_ex['DJDATE_Y_min_RG']=data_ex['DJDATE_Y_min']-data_ex['RGYEAR']
data_ex.drop(['DJDATE_Y_max', 'DJDATE_Y_min'],axis=1,inplace=True)

#df7.columns
data_ex['LAWDATE_Y_max_RG']=data_ex['LAWDATE_Y_max']-data_ex['RGYEAR']
data_ex['LAWDATE_Y_min_RG']=data_ex['LAWDATE_Y_min']-data_ex['RGYEAR']
data_ex.drop(['LAWDATE_Y_max', 'LAWDATE_Y_min'],axis=1,inplace=True)

#df8.columns
data_ex['breakfaith_FBY_max_RG']=data_ex['breakfaith_FBY_max']-data_ex['RGYEAR']
data_ex['breakfaith_FBY_min_RG']=data_ex['breakfaith_FBY_min']-data_ex['RGYEAR']
data_ex.drop(['breakfaith_FBY_max', 'breakfaith_FBY_min'],axis=1,inplace=True)

#df9.columns
data_ex['RECDATE_Y_max_RG']=data_ex['RECDATE_Y_max']-data_ex['RGYEAR']
data_ex['RECDATE_Y_min_RG']=data_ex['RECDATE_Y_min']-data_ex['RGYEAR']
data_ex.drop(['RECDATE_Y_max', 'RECDATE_Y_min'],axis=1,inplace=True)


#df10.columns
data_ex['QBEGIN_Y_max_RG']=data_ex['QBEGIN_Y_max']-data_ex['RGYEAR']
data_ex['QBEGIN_Y_min_RG']=data_ex['QBEGIN_Y_min']-data_ex['RGYEAR']
data_ex.drop(['QBEGIN_Y_max', 'QBEGIN_Y_min'],axis=1,inplace=True)

data_ex=data_ex.drop('RGYEAR',axis=1)

data_ex.to_csv('C:/Luoshichao/ccf_4/quarter-final/input/processedData/feature_ex.csv',index=None,encoding='utf-8')









