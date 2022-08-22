# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:04:51 2022

@author: diosimarcardoza
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import load_model
import keras.backend as K

##################################
# EVALUATE ON TEST DATA - regression model
##################################

# Path from where to retrieve the model output file
output_path = 'model/regression_model_v1.h5'
sequence_length = 50

test_data = pd.read_csv("Data/test.csv")

n_turb = test_data['id'].unique().max()

# pick the feature columns 
sensor_cols = ['s_' + str(i) for i in range(0,21)]
sequence_cols = ['setting_0', 'setting_1', 'setting_2', 'cycle_norm']
sequence_cols.extend(sensor_cols)


# We pick the last sequence for each id in the test data
seq_array_test_last = [test_data[test_data['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in range(1, n_turb + 1) if len(test_data[test_data['id']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

print("This is the shape of the test set: {} turbines, {} cycles and {} features.".format(
    seq_array_test_last.shape[0], seq_array_test_last.shape[1], seq_array_test_last.shape[2]))

print("There is only {} turbines out of {} as {} turbines didn't have more than {} cycles.".format(
    seq_array_test_last.shape[0], n_turb, n_turb - seq_array_test_last.shape[0], sequence_length))

# Selecting and reshaping the labels
y_mask = [len(test_data[test_data['id']==id]) >= sequence_length for id in test_data['id'].unique()]
label_array_test_last = test_data.groupby('id')['RUL'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

#________________________________________________________________
## Predicting the RUL for test data
import keras
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# if best iteration's model was saved then load and use it
if os.path.isfile(output_path):
    #estimator = load_model(output_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
    estimator = load_model(output_path, custom_objects={'root_mean_squared_error': keras.metrics.RootMeanSquaredError()})
    

    y_pred_test = estimator.predict(seq_array_test_last)
    y_true_test = label_array_test_last
    
    # test metrics
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
    print('\nMSE: {}'.format(scores_test[0]))
    print('\nMSE: {}'.format(scores_test[1]))
    print('\nMAE: {}'.format(scores_test[2]))
    
    s1 = ((y_pred_test - y_true_test)**2).sum()
    moy = y_pred_test.mean()
    s2 = ((y_pred_test - moy)**2).sum()
    s = 1 - s1/s2
    print('\nEfficiency: {}%'.format(s * 100))

    test_set = pd.DataFrame(y_pred_test)


if os.path.isfile(output_path):
    # Plot in blue color the predicted data and in green color the
    # actual data to verify visually the accuracy of the model.
    fig_verify = plt.figure(figsize=(60, 30))
    # plt.plot(y_pred_test, 'ro', color="red", lw=3.0)
    # plt.plot(y_true_test, 'ro', color="blue")
    X = np.arange(1, 94)
    width = 0.35
    plt.bar(X, np.array(y_pred_test).reshape(93,), width, color='r')
    plt.bar(X + width, np.array(y_true_test).reshape(93,), width, color='b')
    plt.xticks(X)
    plt.title('Remaining Useful Life for each turbine')
    plt.ylabel('RUL')
    plt.xlabel('Turbine')
    plt.legend(['predicted', 'actual data'], loc='upper left')
    plt.show()
    
    #####################################
    
    
    
##################################
# EVALUATE ON TEST DATA - binary classification model
##################################


# We pick the last sequence for each id in the test data

seq_array_test_last = [test_data[test_data['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in test_data['id'].unique() if len(test_data[test_data['id']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
print("seq_array_test_last")
print(seq_array_test_last)
print(seq_array_test_last.shape)

# Similarly, we pick the labels

#print("y_mask")
# serve per prendere solo le label delle sequenze che sono almeno lunghe 50
y_mask = [len(test_data[test_data['id']==id]) >= sequence_length for id in test_data['id'].unique()]
print("y_mask")
print(y_mask)
label_array_test_last = test_data.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
print(label_array_test_last.shape)
print("label_array_test_last")
print(label_array_test_last)

# if best iteration's model was saved then load and use it
model_path = 'model/binary_model.h5' 
if os.path.isfile(model_path):
    estimator = load_model(model_path)

# test metrics
scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
print('Accurracy: {}'.format(scores_test[1]))

# make predictions and compute confusion matrix
y_pred_test =  np.round(estimator.predict(seq_array_test_last))
y_true_test = label_array_test_last

test_set = pd.DataFrame(y_pred_test)
#test_set.to_csv('../../Output/binary_submit_test.csv', index = None)

from sklearn.metrics import confusion_matrix, recall_score, precision_score
print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true_test, y_pred_test)
print(cm)

# compute precision and recall
precision_test = precision_score(y_true_test, y_pred_test)
recall_test = recall_score(y_true_test, y_pred_test)
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
print( 'Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test )

# Plot in blue color the predicted data and in green color the
# actual data to verify visually the accuracy of the model.
fig_verify = plt.figure(figsize=(100, 50))
plt.plot(y_pred_test, color="blue")
plt.plot(y_true_test, color="green")
plt.title('prediction')
plt.ylabel('value')
plt.xlabel('row')
plt.legend(['predicted', 'actual data'], loc='upper left')
plt.show()
fig_verify.savefig("../../Output/model_verify.png")