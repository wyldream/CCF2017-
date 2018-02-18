import numpy as np
import pandas as pd
import tables

data_path = "../../input/"

entbase_data = pd.read_hdf(data_path + '/processedData/entbase.h5')
alter_data = pd.read_hdf(data_path + 'processedData/alter.h5').drop('ENDDATE',1).drop('TARGET',1)
branch_data = pd.read_hdf(data_path + 'processedData/branch.h5').drop('ENDDATE',1).drop('TARGET',1)
invest_data = pd.read_hdf(data_path + 'processedData/invest.h5').drop('ENDDATE',1).drop('TARGET',1)
right_data = pd.read_hdf(data_path + 'processedData/right.h5').drop('ENDDATE',1).drop('TARGET',1)
project_data = pd.read_hdf(data_path + 'processedData/project.h5').drop('ENDDATE',1).drop('TARGET',1)
lawsuit_data = pd.read_hdf(data_path + 'processedData/lawsuit.h5').drop('ENDDATE',1).drop('TARGET',1)
breakfaith_data = pd.read_hdf(data_path + 'processedData/breakfaith.h5').drop('ENDDATE',1).drop('TARGET',1)
recruit_data = pd.read_hdf(data_path + 'processedData/recruit.h5').drop('ENDDATE',1).drop('TARGET',1)
qualification_data = pd.read_hdf(data_path + 'processedData/qualification.h5').drop('ENDDATE',1).drop('TARGET',1)

all_data = entbase_data                                                 # branch_data,invest_data,right_data,recruit_data,qualification_data,lawsuit_data，project_data,breakfaith_data,
for data in [alter_data,branch_data,invest_data,right_data,recruit_data,qualification_data,lawsuit_data,project_data,breakfaith_data,]:
    all_data = pd.merge(all_data,data,'left','EID')

print(all_data)

all_data.to_hdf(data_path + '/processedData/all_data.h5', 'w', complib='blosc', complevel=5)
# all_data.to_csv(data_path + '/processedData/all_data.csv', index = None,)