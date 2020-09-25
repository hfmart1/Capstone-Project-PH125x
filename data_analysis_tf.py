#################################################
## IMPORT LIBRARIES
#################################################

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
print("Working directory", Path().absolute())


import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers

print("Tensorflow version : ", tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

#################################################
## IMPORT AND CLEAN DATA
#################################################

##Import data table in tsv format
dl = pd.read_csv("migr_asydcfsta.tsv", sep='\t')

##Clean labels with csv encoding
dl_temp1 = dl[dl.columns[0]]
dl_temp1 = dl_temp1.str.split(",", expand = True)
dl_temp1.columns = ["unit", "citizen", "sex" , "age", "decision", "geo"]
dl_temp1 = dl_temp1.drop('unit', 1)

##Clean year number labels
dl_temp2 = dl[dl.columns[1:13]]
dl_temp2.columns = ["X2019", "X2018", "X2017", "X2016",
                    "X2015", "X2014", "X2013", "X2012",
                    "X2011", "X2010", "X2009", "X2008"]
dl_temp2 = dl_temp2.replace(":", "0", regex = True)
dl_temp2 = dl_temp2.replace(to_replace = {" b", " e", " p", " u", " z", " d", " o", " bd",
                                          " bde", " bdep", " bdp", " bdu", " bduo", " be",
                                          " bep", " bp", " bu", " buo", " bz"," de", " dep",
                                          " dp", " du", " duo"," ep","p u", " puo"
                                          },
                            value = "",
                            regex = True
                            )
dl_temp2 = dl_temp2.replace(" ", "", regex = True)
dl_temp2 = dl_temp2.astype(int)

####################
## ONE-HOT ENCODING
####################

##Encode each label
one_hot_geo = pd.get_dummies(dl_temp1["geo"], prefix = "geo")
one_hot_cit = pd.get_dummies(dl_temp1["citizen"], prefix = "cit")
one_hot_sex = pd.get_dummies(dl_temp1["sex"], prefix = "sex")
one_hot_dec = pd.get_dummies(dl_temp1["decision"], prefix = "dec")
one_hot_age = pd.get_dummies(dl_temp1["age"], prefix = "age")


##Join the arrays
one_hot_temp1 = one_hot_geo.join(one_hot_cit)
one_hot_temp2 = one_hot_temp1.join(one_hot_sex)
one_hot_temp3 = one_hot_temp2.join(one_hot_dec)
one_hot_temp4 = one_hot_temp3.join(one_hot_age)


dl_temp3 = one_hot_temp3.join(dl_temp2)


#################################################
## PREPARE DATA
#################################################

## Divide train/test
dataset = dl_temp3
print(dataset)

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


## Data stats
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats)


## Split features from labels
train_labels = train_dataset.pop('X2019')
test_labels = test_dataset.pop('X2019')


## Normalize the data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_train_data.drop('X2019', axis=1, inplace=True)
normed_test_data = norm(test_dataset)
normed_test_data.drop('X2019', axis=1, inplace=True)



#################################################
## NEURAL NETWORK APPROACH
#################################################


## Build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()


## Inspect the model
model.summary()


## Train the model
EPOCHS = 100

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data,
                    train_labels,
                    epochs = EPOCHS,
                    validation_split = 0.2,
                    verbose = 0,
                    callbacks = [early_stop,
                                 tfdocs.modeling.EpochDots()]
                    )


##Visualize the model's training progress

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


plt.plot(hist["epoch"], hist["mae"])
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.savefig('epochmae.pdf')
plt.show()

plt.plot(hist["epoch"], hist["mse"], color = "red")
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.savefig('epochmse.pdf')
plt.show()



##Predict the validation labels
test_predictions = model.predict(normed_test_data).flatten()


##Evaluate the model
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print(rmse(test_predictions, test_labels))


##  Model: "sequential"
##  _________________________________________________________________
##  Layer (type)                 Output Shape              Param #   
##  =================================================================
##  dense (Dense)                (None, 64)                17024     
##  _________________________________________________________________
##  dense_1 (Dense)              (None, 64)                4160      
##  _________________________________________________________________
##  dense_2 (Dense)              (None, 1)                 65        
##  =================================================================
##  Total params: 21,249
##  Trainable params: 21,249
##  Non-trainable params: 0
##  _________________________________________________________________
##
##  Epoch: 0, loss:891718.3750,  mae:31.7619,  mse:891718.3750,  val_loss:173475.3594,  val_mae:25.2753,  val_mse:173475.3594,  
##  ......................................................................
##
##  RMSE : 174.7941588237991




##Interesting Plots
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [X2019]')
plt.ylabel('Predictions [X2019]')
lims = [0, 400000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims, color = "red")
plt.savefig('truepredvalues.pdf')
plt.show()


error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.yscale("log")
plt.savefig('error.pdf')
plt.show()

