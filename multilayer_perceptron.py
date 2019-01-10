from __future__ import print_function

import sys
import time
import subprocess
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 12

def create_preprocess_data():
    
  # the data, split between train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # Transform the image data to fix the training 
  x_train = x_train.reshape(60000, 784) # Train set contains 2D arrays of 28x28 pixels, so reshare them to 1D
  x_test = x_test.reshape(10000, 784)
  x_train = x_train.astype('float32') # float32 is convinient for the neutral network
  x_test = x_test.astype('float32')
  x_train /= 255 # Normalize of the values to the range of [0,1]
  x_test /= 255

  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices, so that in the outer layer we have the possibilities of each class
  # For example, class 5 is transformed in the following matrix: [ 0 0 0 0 0 1 0 0 0 0 ]
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  return (x_train, y_train), (x_test, y_test)

def create_network(hidden_layer_num, first_layer_act_func, hidden_layer_act_func, last_layer_act_func):
    
  model = Sequential()

  # First layer has an input of 784, the 1D feature vector 
  model.add(Dense(512, activation=first_layer_act_func, input_shape=(784,)))

  hidden_layers_range = range(hidden_layer_num)

  for n in hidden_layers_range:
    model.add(Dropout(0.2))
    model.add(Dense(256, activation=hidden_layer_act_func))
  
  # last classification layer
  model.add(Dense(num_classes, activation=last_layer_act_func))

  model.summary() 

  model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
  
  return model

    
def train_model_and_evaluate(model, x_train, y_train, x_test, y_test):

  start = time.time()
  history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
  end = time.time()
  duration = end - start
  score = model.evaluate(x_test, y_test, verbose=0)

  accuracy = score[1]
  
  return (duration,accuracy)
    
def main():
  
  # Parse the arguments
  hidden_layers_num_str = sys.argv[1]
  first_layer_activation_func = sys.argv[2]
  hidden_layers_activation_func = sys.argv[3]
  last_layer_activation_func = sys.argv[4]
  
  # Create data
  (x_train, y_train), (x_test, y_test) = create_preprocess_data()
  
  # Create MLP
  hidden_layers_num = int(hidden_layers_num_str)
  model = create_network(hidden_layers_num, first_layer_activation_func, hidden_layers_activation_func, last_layer_activation_func)

  # Train and evaluate
  (duration,accuracy) = train_model_and_evaluate(model, x_train, y_train, x_test, y_test)
  
  # Save the results in txt
  result_hidden_layers = "\n------- \nHidden layer number: %d" % hidden_layers_num
  result_duration = "\nDuration: %d seconds" % duration
  result_accuracy = "\nAccuracy: %.4f " % accuracy
  result_activation_first = "\nFirst activation func: " + first_layer_activation_func
  result_activation_hidden = "\nHidden activation func: " + hidden_layers_activation_func
  result_activation_last = "\nLast activation func " + last_layer_activation_func
  result = result_hidden_layers + result_duration + result_accuracy + result_activation_first + result_activation_hidden + result_activation_last

  with open("results.txt", "a") as myfile:
    myfile.write(result)
  myfile.close()  
if __name__ == "__main__":
    main()
