from __future__ import print_function

import sys
import time
import subprocess
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from rbflayer import RBFLayer, InitCentersRandom, InitCentersKMeans

class TrainConfig:

    def __init__(self, epochs, batch_size, use_kmeans, k_num, random_samples_num, betas, dropoutRate, hidden_layers_num, hidden_layer_act_func, last_layer_act_func):
        self.use_kmeans = use_kmeans
        self.k_num = k_num
        self.random_samples_num = random_samples_num
        self.batch_size = batch_size
        self.epochs = epochs
        self.betas = betas
        self.dropoutRate = dropoutRate
        self.hidden_layers_num = hidden_layers_num
        self.hidden_layer_act_func = hidden_layer_act_func
        self.last_layer_act_func = last_layer_act_func

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
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

# Prepares the network with the provided paramaters
def create_network(train_set, trainingConfig):

    model = Sequential()

    # Create the RBF layer
    if trainingConfig.use_kmeans:
        initializer = InitCentersKMeans(train_set, trainingConfig.k_num)
        layer_exit_num = trainingConfig.k_num
    else:
        initializer = InitCentersRandom(train_set, trainingConfig.random_samples_num)
        layer_exit_num = trainingConfig.random_samples_num

    rbflayer = RBFLayer(layer_exit_num,
                        initializer,
                        betas = trainingConfig.betas,
                        input_shape=(784,))

    # First layer is the RBF layer
    model.add(rbflayer)

    if trainingConfig.hidden_layers_num > 0:

        if trainingConfig.use_kmeans:
            hidden_layer_output = trainingConfig.k_num
        else:
            hidden_layer_output = trainingConfig.random_samples_num

        # Add the hidden layers, Dense neural is used in combination with Dropout
        hidden_layers_range = range(trainingConfig.hidden_layers_num)

        for n in hidden_layers_range:
            model.add(Dropout(trainingConfig.dropoutRate))
            model.add(Dense(units=hidden_layer_output, activation=trainingConfig.hidden_layer_act_func))

    # last classification layer, output dim is 10, 10 possible classes
    model.add(Dense(units=10, activation=trainingConfig.last_layer_act_func))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(), # Used for multiclass problems
                  metrics=['accuracy']) # Accuracy because the problem solved is classification

    return (model, rbflayer)

def store_results(rbflayer, trainingConfig, train_duration, test_duration, train_accuracy, test_accuracy):

    if trainingConfig.use_kmeans:
        training_way = 'K_means'
        starting_centers = trainingConfig.k_num
    else:
        training_way = 'Random_Centers'
        starting_centers = trainingConfig.random_samples_num

    intro = "\n-----------------\n"

    train_params = "Train parameters:\n Epochs:{epo}\n Batch size: {b_size}\n Training way: {train_way}\n Initial centers num: {init_centers_num}\n rbf_betas: {rbf_betas}\n Dropout rate: {drop_rate}\n Hidden layers num: {hid_lay_num}\n Hidden layer activ function:{hid_lay_act_func}\n Last layer activ function:{last_lay_act_func}".format(epo=trainingConfig.epochs, b_size=trainingConfig.batch_size, train_way=training_way, init_centers_num=starting_centers, rbf_betas=trainingConfig.betas, drop_rate=trainingConfig.dropoutRate, hid_lay_num=trainingConfig.hidden_layers_num, hid_lay_act_func=trainingConfig.hidden_layer_act_func, last_lay_act_func=trainingConfig.last_layer_act_func)
    duration_train = "\nTrain duration %.2f seconds" % train_duration
    duration_test = "\nTest duration %.2f seconds" % test_duration
    train_accuracy = "\nTrain Accuracy: {acc}".format(acc=train_accuracy)
    test_accuracy = "\nTest Accuracy: {acc}".format(acc=test_accuracy)

    result = intro + train_params + duration_train + duration_test + train_accuracy + test_accuracy

    with open("rbf_network_results.txt","a") as output_file:
        output_file.write(result)
    output_file.close()

def train_test_model(rbflayer, model, trainingConfig, train_set, train_labels, test_set, test_labels):

    start_time_train = time.time()

    hist = model.fit(train_set,
              train_labels,
              epochs=trainingConfig.epochs,
              verbose=1)

    end_time_train = time.time()
    train_duration = end_time_train - start_time_train
    train_acc = hist.history['acc']

    start_time_test = time.time()
    test_accuracy = model.evaluate(test_set, test_labels)[1]
    end_time_test = time.time()

    test_duration = end_time_test - start_time_test

    store_results(rbflayer, trainingConfig, train_duration, test_duration, train_acc, test_accuracy)

if __name__ == "__main__":

    (train_set, train_labels), (test_set, test_labels) = create_preprocess_data()

    # Using random centers

    # Different samples number
    t1 = TrainConfig(epochs=15, batch_size=128, use_kmeans=False, k_num=0, random_samples_num=60, betas=2.0, dropoutRate= 0.1, hidden_layers_num=0, hidden_layer_act_func='relu', last_layer_act_func='softmax')
    t2 = TrainConfig(epochs=15, batch_size=128, use_kmeans=False, k_num=0, random_samples_num=200, betas=2.0, dropoutRate= 0.1, hidden_layers_num=0, hidden_layer_act_func='relu', last_layer_act_func='softmax')
    t10 = TrainConfig(epochs=20, batch_size=128, use_kmeans=False, k_num=0, random_samples_num=30, betas=2.0, dropoutRate= 0.1, hidden_layers_num=1, hidden_layer_act_func='relu', last_layer_act_func='softmax')
    t13 = TrainConfig(epochs=30, batch_size=256, use_kmeans=False, k_num=0, random_samples_num=600, betas=2.0, dropoutRate= 0.2, hidden_layers_num=1, hidden_layer_act_func='relu', last_layer_act_func='softmax')

    # Using kmeans
    t14 = TrainConfig(epochs=15, batch_size=128, use_kmeans=True, k_num=30, random_samples_num=0, betas=2.0, dropoutRate= 0.1, hidden_layers_num=0, hidden_layer_act_func='relu', last_layer_act_func='softmax')
    t15 = TrainConfig(epochs=15, batch_size=128, use_kmeans=True, k_num=60, random_samples_num=0, betas=3.0, dropoutRate= 0.2, hidden_layers_num=0, hidden_layer_act_func='relu', last_layer_act_func='softmax')
    t16 = TrainConfig(epochs=20, batch_size=128, use_kmeans=True, k_num=120, random_samples_num=0, betas=2.0, dropoutRate= 0.1, hidden_layers_num=1, hidden_layer_act_func='relu', last_layer_act_func='softmax')
    t17 = TrainConfig(epochs=30, batch_size=128, use_kmeans=True, k_num=80, random_samples_num=0, betas=2.0, dropoutRate= 0.1, hidden_layers_num=2, hidden_layer_act_func='relu', last_layer_act_func='softmax')

    (net1, rbf1) = create_network(train_set, t1)
    train_test_model(rbf1, net1, t1, train_set, train_labels, test_set, test_labels)

    (net2, rbf2) = create_network(train_set, t2)
    train_test_model(rbf2, net2, t2, train_set, train_labels, test_set, test_labels)

    (net3, rbf3) = create_network(train_set, t10)
    train_test_model(rbf3, net3, t10, train_set, train_labels, test_set, test_labels)

    (net6, rbf6) = create_network(train_set, t13)
    train_test_model(rbf6, net6, t13, train_set, train_labels, test_set, test_labels)

    (net7, rbf7) = create_network(train_set, t14)
    train_test_model(rbf7, net7, t14, train_set, train_labels, test_set, test_labels)

    (net8, rbf8) = create_network(train_set, t15)
    train_test_model(rbf8, net8, t15, train_set,train_labels, test_set, test_labels)

    (net9, rbf9) = create_network(train_set, t16)
    train_test_model(rbf9, net9, t16, train_set, train_labels, test_set, test_labels)

    (net10, rbf10) = create_network(train_set, t17)
    train_test_model(rbf10, net10, t17, train_set, train_labels, test_set, test_labels)
