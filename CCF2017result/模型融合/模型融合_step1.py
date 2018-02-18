
import pandas as pd
lgb_2=pd.read_csv('E:/CCF2017/result/lgb_6937.csv')
lgb_1=pd.read_csv('E:/company_risk/result/lgb_1.csv')

data=pd.merge(lgb_2,lgb_1,on='EID')

data['PROB']=0.65*data['PROB_x']+0.35*data['PROB_y']

df=data[['EID','PROB']]
df['FORTARGET']=df.PROB.apply(lambda x:1 if x>0.25 else 0)
df1=df[['EID','FORTARGET','PROB']]
df1.to_csv('E:/CCF2017/result_new/lgb_ronghe_6905.csv',index=None,encoding='utf-8')




import pandas as pd
lgb_2=pd.read_csv('E:/CCF2017/result_new/lgb_ronghe_6905.csv')
lgb_1=pd.read_csv('E:/company_risk/result/lgb_6900.csv')

data=pd.merge(lgb_2,lgb_1,on='EID')

data['PROB']=0.65*data['PROB_x']+0.35*data['PROB_y']

df=data[['EID','PROB']]
df['FORTARGET']=df.PROB.apply(lambda x:1 if x>0.25 else 0)
df1=df[['EID','FORTARGET','PROB']]
df1.to_csv('E:/CCF2017/result_new/lgb_6912.csv',index=None,encoding='utf-8')




import pandas as pd
lgb_2=pd.read_csv('E:/CCF2017/result_new/lgb_6912.csv')
lgb_1=pd.read_csv('E:/company_risk/result/lgb_gbdt.csv')

data=pd.merge(lgb_2,lgb_1,on='EID')

data['PROB']=0.65*data['PROB_x']+0.35*data['PROB_y']

df=data[['EID','PROB']]
df['FORTARGET']=df.PROB.apply(lambda x:1 if x>0.22 else 0)
df1=df[['EID','FORTARGET','PROB']]
df1.to_csv('E:/CCF2017/result_new/lgb_6918.csv',index=None,encoding='utf-8')


import pandas as pd
lgb_2=pd.read_csv('E:/CCF2017/result_new/lgb_6918.csv')
lgb_1=pd.read_csv('E:/CCF2017/result/mul_xgb.csv')

re_6954=pd.read_csv('E:/CCF2017/result/6954.csv')
data=pd.merge(lgb_2,lgb_1,on='EID')
data['PROB']=0.65*data['PROB_x']+0.35*data['PROB_y']

df=data[['EID','PROB']]
df['FORTARGET']=re_6954.FORTARGET
df1=df[['EID','FORTARGET','PROB']]
df1.to_csv('E:/CCF2017/1209/ronghe_1209.csv',index=None,encoding='utf-8')