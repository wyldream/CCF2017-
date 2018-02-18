import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


data_path = "../../input/"

def load_data(file_name):
    return pd.read_csv(data_path+file_name)

if __name__ == '__main__':
    print('It is main!!!')
else:
    print('Successfully Import UDFs!!!')
