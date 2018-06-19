

from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import keras as K
from keras.models import Sequential
from keras.layers import Embedding, LSTM, TimeDistributed, Dense


def get_dataset(file_name, one_hot=False):
    """
        Will return the dataset in a trainable format
    :param file_name: name of the file to extract the data from...
    :param one_hot: want the data as one hot or not?
    :return: the addresses along with their hit/miss status
    """
    def fit_to_length(addr, target_length):
        # step 1. make same size
        while len(addr) < target_length:
            addr = '0'+addr
        return addr

    with open(file_name) as this_file:
        examples = [x.split(',') for x in this_file.readlines()]
        dataset = np.asarray(examples)

    if not one_hot:
        return [fit_to_length(x, 10) for x in dataset[:, 0]], np.asarray([map(int, dataset[:, 1])])

    labels = np.asarray(map(int, dataset[:, 1]))
    one_hot_labels = np.zeros(shape=(labels.shape[0], 2))
    one_hot_labels[range(labels.shape[0]), labels] = 1
    return [fit_to_length(x, 10) for x in dataset[:, 0]], np.asarray(one_hot_labels)


class Network(object):
    """
        Defines the lstm network and trains it
    """
    def __init__(self, input_length, embed_dim = 10, lstm_out = 200):
        self.model = Sequential()
        # we won't create embeddings since we already have them in our code
        # self.model.add(Embedding(10, embed_dim, input_length=input_length))
        self.model.add(LSTM(units=lstm_out, input_length=input_length, input_dim=input_length, return_sequences=True))
        # self.model.add(TimeDistributed(Dense(200)))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def train_model(self, train_set, test_set, learning_rate, epochs, batch_size):
        self.model.fit(x=train_set[0], y=train_set[1], batch_size=batch_size, epochs=epochs, validation_data=test_set)

        pass

    def test_model(self):
        pass


def main():
    # we receive the addresses as strings and their hit/miss labels as integers
    train_addresses, train_labels = get_dataset(file_name='train.txt', one_hot=False)
    test_addresses, test_labels = get_dataset(file_name='test.txt', one_hot=False)
    print('log: train data size: {} {}'.format(len(train_addresses), train_labels.shape))
    print('log: test data size: {} {}'.format(len(test_addresses), test_labels.shape))

    # next, we need a routine to convert each address into a sequence of vectors that we could feed into an lstm
    # and also we need to make them the same size as well
    def address_to_vector(addr, target_length):
        # step 1. make same size
        while len(addr) < target_length:
            addr = '0'+addr
        # step 2. turn into a matrix, basically target_length*10, target_length for the digits in the address,
        #         and 10 for the 10 possible digits at each spot, so... (and we assume a batch size = 1)
        vec = np.zeros(shape=(target_length, 1, 10)) # since there are 10 possible digits!
        for index, digit in enumerate(addr):
            vec[index, 0, int(digit)] = 1
        return vec

    int_train_addresses =

    embedded_train = np.asarray([address_to_vector(x, 10) for x in train_addresses]).reshape(-1, 10, 10)
    embedded_test = np.asarray([address_to_vector(x, 10) for x in test_addresses]).reshape(-1, 10, 10)
    print('log: new embedded train addresses size: {}'.format(embedded_train.shape))
    print('log: new embedded test addresses size: {}'.format(embedded_test.shape))
    # print(train_addresses[675], embedded_addresses[675, :, :])

    lstm = Network(input_length=10)
    lstm.train_model(train_set=(embedded_train, train_labels), test_set=(embedded_test, test_labels),
                     learning_rate=1e-3, epochs=5, batch_size=32)
    pass


if __name__ == '__main__':
    main()



