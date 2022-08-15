#Import all required packages and modules 
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.functions import reshapeFeatures, reshapeLabel, root_mean_squared_error
#_____________________________________________________________
######### loading data##########
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')
n_turb = train_data['id'].unique().max()


# pick the feature columns 
sensor_cols = ['s_1','s_2', 's_3', 's_6', 's_7', 's_8','s_10','s_11', 's_12', 's_13', 's_14', 's_16', 's_19', 's_20']
sequence_cols = ['setting_0', 'setting_1', 'setting_2', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# generator for the sequences
sequence_length = 50
feat_gen = (list(reshapeFeatures(train_data[train_data['id']==id], sequence_length, sequence_cols)) 
           for id in range(1, n_turb + 1))

# generate sequences and convert to numpy array
feat_array = np.concatenate(list(feat_gen)).astype(np.float32)


print("The data set has now shape: {} entries, {} cycles and {} features.".format(feat_array.shape[0],
                                                                                  feat_array.shape[1],
                                                                                  feat_array.shape[2]))

# generate labels
label_gen = [reshapeLabel(train_data[train_data['id']==id]) for id in range(1, n_turb + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)
print(label_array.shape)
#__________________________________________________________________________
## Create a Recurrent Neural Network - LSTM
nb_features = feat_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()
model.add(LSTM(input_shape=(sequence_length, nb_features), units=100, return_sequences=True, name="lstm_0"))
model.add(Dropout(0.2, name="dropout_0"))
model.add(LSTM(units=50, return_sequences=True, name="lstm_1"))
model.add(Dropout(0.2, name="dropout_1"))
model.add(LSTM(units=25, return_sequences=False, name="lstm_2"))
model.add(Dropout(0.2, name="dropout_2"))
model.add(Dense(units=nb_out, name="dense_0"))
model.add(Activation("linear", name="activation_0"))
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[root_mean_squared_error, 'mae'])

print(model.summary())
#__________________________________________________________________________
## Train the Recurrent Neural Network - LSTM
output_path = 'model/regression_model_v0.h5'

epochs = 100
batch_size = 200

# fit the network
history = model.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                     verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_loss',
                                                       save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path))
#_______________________________________________________________________________________

# summarize history for MAE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#fig_acc.savefig("output/model_mae.png")

# summarize history for RMSE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model RMSE')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#fig_acc.savefig("output/model_rmse.png")

# summarize history for Loss
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#fig_acc.savefig("output/model_regression_loss.png")
