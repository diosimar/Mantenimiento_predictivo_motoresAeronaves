#Import  all requaried package and modules 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#___________________________________________________________________________________________________________
######### loading data##########
names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
         's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# read training data
train_data = pd.read_csv('input/TrainSet.txt', sep=" ", header=None)
train_data.drop(train_data.columns[[26, 27]], axis=1, inplace=True)
train_data.columns = names

train_data = train_data.sort_values(['id','cycle'])

# read test data
test_data = pd.read_csv('input/TestSet.txt', sep=" ", header=None)
test_data.drop(test_data.columns[[26, 27]], axis=1, inplace=True)
test_data.columns = names

# read ground truth data
truth_df = pd.read_csv('input/TestSet_RUL.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
