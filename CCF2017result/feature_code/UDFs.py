import seaborn as sns
from pandas_summary import DataFrameSummary
import tables
import re

# 定义通用函数
def gr_agg(df, by_name, col_name,  *functions):
    gr = df.groupby(by_name)
    mapper = lambda x: col_name + '_' + x if x != by_name else by_name # col_name_sum x是函数名称（sum、mean……）
    return gr[col_name].agg(functions).reset_index().rename(columns=mapper)


if __name__ == 'main':
    print('This is main')
else:
    print('import UDFs successful')