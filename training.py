#Import all required packages and modules 
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.functions import reshapeFeatures, reshapeLabel, root_mean_squared_error,r2_keras
#_____________________________________________________________
######### loading data##########
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')
n_turb = train_data['id'].unique().max()
["s_" + str(i) for i in range(21)]

# pick the feature columns 
sensor_cols = ["s_" + str(i) for i in range(21)]
#sensor_cols = ['s_1','s_2', 's_3', 's_6', 's_7', 's_8','s_10','s_11', 's_12', 's_13', 's_14', 's_16', 's_19', 's_20']
sequence_cols = ['setting_0', 'setting_1', 'setting_2', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# generator for the sequences
sequence_length = 50

feat_gen = (list(reshapeFeatures(train_data[train_data['id']==id], sequence_length, sequence_cols)) 
           for id in range(1, n_turb + 1))

# val is a list of 192 - 50 = 142 bi-dimensional array (50 rows x 25 columns)
val=list(reshapeFeatures(train_data[train_data['id']==1], sequence_length, sequence_cols))
print(len(val))

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
## Create a Recurrent Neural Network - LSTM (Regression)
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
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[keras.metrics.RootMeanSquaredError(),keras.metrics.MeanAbsolutePercentageError(), 'mae'])

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


# training metrics
scores = model.evaluate(feat_array, label_array, verbose=1, batch_size=200)
print('\nMAE: {}'.format(scores[1]))
print('\nR^2: {}'.format(scores[2]))
#_______________________________________________________________________________________


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


# summarize history for R^2
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('model r^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

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

#__________________________________________________________________________
## Create a Recurrent Neural Network - LSTM (binary_classification)

from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import load_model

model_path = 'model/binary_model.h5'
# generate labels 
label_gen = [reshapeLabel(train_data[train_data['id']==id], label=['label1']) for id in range(1, n_turb + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)
print(label_array.shape)


model = Sequential()

model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


# fit the network
history = model.fit(feat_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())

# summarize history for Accuracy
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("../../Output/model_accuracy.png")

# summarize history for Loss
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("../../Output/model_loss.png")

# training metrics
scores = model.evaluate(feat_array, label_array, verbose=1, batch_size=200)
print('Accurracy: {}'.format(scores[1]))

# make predictions and compute confusion matrix
y_pred = np.round(model.predict(feat_array,verbose=1, batch_size=200))
y_true = label_array

print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true, y_pred)
print(cm)
# compute precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print( 'precision = ', precision, '\n', 'recall = ', recall)

'''
one_engine = []
health_condition = []
for i, r in train_data.iterrows():
   rul = r.RUL
   hc =  r.health_condition
   one_engine.append(rul)
   health_condition.append(hc)
   if rul == 0:
     plt.plot(one_engine, health_condition )
     one_engine =[]
     health_condition = []
plt.grid()

df = []
for i in train_data.id.unique():
    prueba = train_data[train_data.id == i]
    prueba['health_condition'] = prueba.RUL / max(prueba.RUL)
    df.append(prueba)
    
df = pd.concat(df)
    
plt.plot(prueba.cycle, prueba.health_condition)

preuba = prueba[prueba.id == 1]


plt.plot(prueba.RUL)
for i, r in train_data.iterrows():
    print(r)


plt.plot(train_data[train_data["id"] ==1].cycle, train_data[train_data["id"] == 1]['health_condition'])
'''''