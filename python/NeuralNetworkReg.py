from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

def build_model():
  model = keras.Sequential([
    layers.Dense(2, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(50, activation=tf.nn.relu),
    layers.Dense(50, activation=tf.nn.relu),	
    layers.Dense(1)
  ])


  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')



def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,1000])
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,1000])
  plt.show()



#column_names = ['Wing area','Fuel weight','Aspect ratio','Quarter chord sweep','Dynamic pressure',
#                'Taper ratio', 'Airfoil thickness to chord ratio', 'Ultimate load factor','Flight design weight', 'Paint weight', 'Wing weight' ]

column_names = ['x1','x2', 'Wing weight' ]


dataset = pd.read_csv(str(sys.argv[1]),names=column_names)

#print(str(sys.argv[1]))

#print(dataset)


train_dataset = dataset.sample(frac=0.2,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print('training dataset')
print(train_dataset)
print('test dataset')
print(test_dataset)





#sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
#plt.show()


train_stats = train_dataset.describe()
train_stats.pop("Wing weight")
train_stats = train_stats.transpose()
#print(train_stats)

train_labels = train_dataset.pop('Wing weight')
test_labels = test_dataset.pop('Wing weight')

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)





print(train_labels)


#print(normed_train_data)


model = build_model()

#print(model.summary())




#example_batch = normed_train_data[:10]
#example_result = model.predict(example_batch)
#print(example_result)



EPOCHS = 10000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



#history = model.fit(normed_train_data, train_labels,epochs=EPOCHS, validation_split = 0.2, verbose=0,callbacks=[PrintDot()])


history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


#plot_history(history)


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))
print("Testing set Mean Sqr Error: {:5.2f}".format(mse))


# Open a file
fo = open("NNoutput.dat", "w")
fo.write( "{:5.2f}\n".format(mae));
fo.write( "{:5.2f}\n".format(mse));
# Close opend file
fo.close()

test_predictions = model.predict(normed_test_data).flatten()

print(test_predictions)


#plt.scatter(test_labels, test_predictions)
#plt.xlabel('True Values [Wing weight]')
#plt.ylabel('Predictions [Wing weight]')
#plt.axis('equal')
#plt.axis('square')
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-100, 600], [-100, 600])
#plt.show()




