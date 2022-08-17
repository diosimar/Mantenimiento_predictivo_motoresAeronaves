#Import all required packages and modules 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#___________________________________________________________________________________________________________
######### loading data##########
names = ["id", "cycle"] + ["setting_" + str(i) for i in range(3)] + ["s_" + str(i) for i in range(21)]

# read training data
train_data = pd.read_csv('Data/TrainSet.txt', sep=" ", header=None)
train_data.drop(train_data.columns[[26, 27]], axis=1, inplace=True)
train_data.columns = names

train_data = train_data.sort_values(['id','cycle'])

# read test data
test_data = pd.read_csv('Data/TestSet.txt', sep=" ", header=None)
test_data.drop(test_data.columns[[26, 27]], axis=1, inplace=True)
test_data.columns = names
test_data = test_data.sort_values(['id','cycle'])

# read ground truth data
truth_df = pd.read_csv('Data/TestSet_RUL.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

'''
cols = ['id', 'cycle', 'setting_0', 'setting_1', 'setting_2','s_1','s_2', 's_3', 's_6', 's_7', 's_8','s_10',
       's_11', 's_12', 's_13', 's_14', 's_16', 's_19', 's_20']

train_data = train_data[cols]
test_data = test_data[cols]
'''

#___________________________________________________________________________

n_turb = train_data["id"].unique().max()
n_train, n_features = train_data.shape
#####_____train set_____#####
# Data Labeling - generate column RUL
rul = pd.DataFrame(train_data.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_data = train_data.merge(rul, on=['id'], how='left')
train_data['RUL'] = train_data['max'] - train_data['cycle']
train_data.drop('max', axis=1, inplace=True)

df = []
for i in train_data.id.unique():
    data_ = train_data[train_data.id == i]
    data_['health_condition'] = data_.RUL / max(data_.RUL)
    df.append(data_)
    
train_data = pd.concat(df)

# generate label columns for training data
# we will only make use of "label1" for binary classification, 
# while trying to answer the question: is a specific engine going to fail within w1 cycles?
# theses values  can be a  parameters to set the windows for classification, this depends on what time window the client wants to specify 
w1 = 30
w0 = 15
train_data['label1'] = np.where(train_data['RUL'] <= w1, 1, 0 )
train_data['label2'] = train_data['label1']
train_data.loc[train_data['RUL'] <= w0, 'label2'] = 2

## we need to scale the variables to control the performance
# MinMax normalization (from 0 to 1)

train_data['cycle_norm'] = train_data['cycle']
cols_normalize = train_data.columns.difference(['id','cycle','RUL','label1','label2','health_condition' ])
min_max_scaler = MinMaxScaler()
norm_train_data = pd.DataFrame(min_max_scaler.fit_transform(train_data[cols_normalize]),
                               columns=cols_normalize, index=train_data.index)
join_data = train_data[train_data.columns.difference(cols_normalize)].join(norm_train_data)
train_data = join_data.reindex(columns = train_data.columns)



#save the training data processed
train_data.to_csv('Data/train.csv', encoding='utf-8',index = None)

#####_____test set_____#####
# MinMax normalization (from 0 to 1)
test_data['cycle_norm'] = test_data['cycle']
norm_test_data = pd.DataFrame(min_max_scaler.transform(test_data[cols_normalize]),
                              columns=cols_normalize, index=test_data.index)
test_join_data = test_data[test_data.columns.difference(cols_normalize)].join(norm_test_data)
test_data = test_join_data.reindex(columns = test_data.columns)
test_data = test_data.reset_index(drop=True)

# generate RUL
rul = pd.DataFrame(test_data.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)

test_data = test_data.merge(truth_df, on=['id'], how='left')
test_data['RUL'] = test_data['max'] - test_data['cycle']
test_data.drop('max', axis=1, inplace=True)

df = []
for i in test_data.id.unique():
    data_ = test_data[test_data.id == i]
    data_['health_condition'] = data_.RUL / max(data_.RUL)
    df.append(data_)
    
test_data = pd.concat(df)

# generate label columns w0 and w1 for test data
test_data['label1'] = np.where(test_data['RUL'] <= w1, 1, 0 )
test_data['label2'] = test_data['label1']
test_data.loc[test_data['RUL'] <= w0, 'label2'] = 2

test_data.to_csv('Data/test.csv', encoding='utf-8',index = None)
