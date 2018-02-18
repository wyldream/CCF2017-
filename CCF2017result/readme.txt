1、数据源（特征）主要说明如下：
(1)特征代码：
依次执行/CCF2017result/features下的
entbase.py、alter.py、branch.py、invest.py、right.py、project.py、lawsuti.py、breakfaith.py、
recruit.py、qualification.py文件可以得到各个表单独构建出的特征，运行all-data.py
会将上述单表特征整合到一起，运行combination.py可以添加组合特征。
上述代码运行出来的h5文件存在与/CCF2017result/data/all_data_1.h5 
(2)特征结果说明
/CCF2017result/data
first_feature: [all_data.h5,feature_ex,[a,b]]
second_feature: [all_data_1.h5,feature_ex,[a,b]]
其中all_data.h5为392维原始特征；feature_ex是29维跨表特征；[a,b]就500维gbdt特征（由gbdt（[all_data.h5,feature_ex]）生成）代码在lgb_gbdt中可以复现
all_data_1.h5为413维原始特征；和all_data.h5主要区别是增加了基本表的衍生特征和一些重要的数值型组合特征
备注：/CCF2017result/data中有个all_data.csv这个文件和all_data_1.h5几乎完全一致，有新增1,2个变量。

2、模型说明：
模型运行顺序：
lgb：
单模型0.6937.py 得 /CCF2017result/middle_result/lgb_6937.csv
lgb_1.py 得 /CCF2017result/middle_result/lgb_1.csv
lgb_ex.py 得 /CCF2017result/middle_result/lgb_6900.csv
lgb_gbdt.py 得 /CCF2017result/middle_result/lgb_gbdt.csv
多xgb：
mul_xgb.py 得 /CCF2017result/middle_result/mul_xgb.csv(6944)
gbdt:
GBDT.py 得 /CCF2017result/middle_result/gbdt6704.csv
rf:
RF.py 得 /CCF2017result/middle_result/rf6662.csv


3、模型融合说明：
模型融合_step1:
result=0.65*lgb_6937+0.35*lgb_1.csv
result=0.65*result+0.35*lgb_6900.csv
result=0.65*result+0.35*lgb_gbdt.csv
ronghe_1209=0.65*result+0.35*mul_xgb.csv  (6958)
备注：模型融合中间结果存于/CCF2017result/result_new/
模型融合_step2:
final = 0.75*ronghe_1209.PROB + 0.15*gbdt6704.PROB + 0.1*rf6662.PROB(6961)
模型融合最终结果存于/CCF2017result/result_end/final.csv

