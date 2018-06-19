

"""
    A simple script to train a small lstm network on memory addresses against their cache hit and miss
    Basically each address is read from left to right using an lstm network, and based on the final state,
    we predict whether it will be a cache hit or miss!
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.optim as optimizers
from torch.nn import functional as F


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
        return [fit_to_length(x, 10) for x in dataset[:, 0]], Tensor([map(int, dataset[:, 1])])

    labels = np.asarray(map(int, dataset[:, 1]))
    one_hot_labels = np.zeros(shape=(labels.shape[0], 2))
    one_hot_labels[range(labels.shape[0]), labels] = 1
    return [fit_to_length(x, 10) for x in dataset[:, 0]], Tensor(one_hot_labels)


class LSTM_net(nn.Module):
    """
        Creates the network, trains it, periodically evaluates it, and finally tests it
    """
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(LSTM_net, self).__init__()
        # self.embeddings = nn.Embedding(total_digits_in_one_address, embedding_dim)
        # lstms in pytorch take in data of size embedding_dim and output their hidden states of size hidden_dim
        # self.hidden_size = hidden_size
        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

        self.embeddings = nn.Embedding(input_size, embedding_size)
        self.input_size = input_size
        self.hidden_dim = hidden_size
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=2)
        # The linear layer that maps from hidden state space to tag space
        self.fc_1 = nn.Linear(hidden_size, 64)
        self.fc_2 = nn.Linear(64, output_size)
        # self.hitormiss = nn.Linear(64, output_size)
        self.hidden = self.initHidden()
        pass

    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))

    def forward(self, address_digit, done):

        # this_embeddings = self.embeddings(Tensor([int(address_digit)], dtype=torch.long))
        # print(this_embeddings.size())
        lstm_out, self.hidden = self.lstm(done.view(1, 1, -1), self.hidden)
        # print(self.hidden)
        # print('lstm', lstm_out)
        fc_1 = self.fc_1(lstm_out.view(-1,self.hidden_dim))
        fc_2 = self.fc_2(fc_1)
        # print(fc_2.size())
        hitormiss = F.log_softmax(fc_2, dim=1)
        # print(hitormiss)
        return hitormiss


    def train_model(self, train_data, test_data, embeddings_function, epochs, learning_rate, log_after=1000):
        self.embeddings_function = embeddings_function
        optimizer = optimizers.SGD(self.parameters(), lr=learning_rate, momentum=0.5)
        example_addresses, labels = train_data
        criterion = nn.NLLLoss()
        # ckpt = torch.load('models/model-1.ckpt')
        # self.load_state_dict(ckpt)
        # print('loaded saved model')
        # print('\n testing now... \n')
        # self.test_model(test_examples=test_data[0], labels=test_data[1], criterion=criterion)
        self.train()
        for e in range(1, epochs+1):
            batch_train_acc = 0
            batch_loss = 0
            for i in range(len(example_addresses)):
                this_idx = int(torch.LongTensor(1).random_(0, len(example_addresses)))
                this_label = torch.tensor([labels[0][this_idx]], dtype=torch.long)
                # zero out the gradients saved previously
                optimizer.zero_grad()
                self.zero_grad()

                # get the embeddings for the address and initiate a zero hidden state
                # vec_of_addr = embeddings_function(example_addresses[this_idx], 10)
                self.hidden = self.initHidden()

                full_emb = embeddings_function(example_addresses[i], 10)
                # pass in each input address digit by digit
                for k in range(len(example_addresses[i])):
                    output = self.forward(example_addresses[this_idx][k], full_emb[k][:])
                prediction = output.max(1, keepdim=True)[1]
                acc = prediction.eq(this_label.view_as(prediction)).sum().item()
                batch_train_acc += acc
                # calculate the loss etc.
                loss = criterion(output, this_label)
                batch_loss += loss.item()
                # print(output, this_label)
                loss.backward()
                optimizer.step()
                # Add parameters' gradients to their values, multiplied by learning rate
                # for p in self.parameters():
                #     print(p.grad)
                #     p.data.add_(-learning_rate, p.grad.data)
                # print(output)
                if i % log_after == 0:
                    print('epoch ({}/{}) progress ({}/{} = {:.0f}%), batch_loss = {:.2f}, '
                          'batch_acc = {:.2f}%'.format(e, epochs, i, len(example_addresses),
                                                      100.0*i/len(example_addresses), batch_loss/(i+1),
                                                      100.0*batch_train_acc/(i+1)))
            print('log: saving model now...')
            torch.save(self.state_dict(), 'models/model-{}.ckpt'.format(e))
            print('\n testing now... \n')
            self.test_model(test_examples=test_data[0], labels=test_data[1], criterion=criterion)
            pass
        pass

    def test_model(self, test_examples, labels, criterion):
        # can test on evaluation or test set
        net_loss, net_acc = (0, 0)
        self.eval()
        with torch.no_grad():
            for i in range(len(test_examples)):
                this_label = torch.tensor([labels[0][i]], dtype=torch.long)

                # get the embeddings for the address and initiate a zero hidden state
                # vec_of_addr = self.embeddings_function(test_examples[this_idx], 10)
                self.hidden = self.initHidden()
                full_emb = self.embeddings_function(test_examples[i], 10)

                # pass in each input address digit by digit
                for k in range(len(test_examples[i])):
                    output = self.forward(test_examples[i][k], full_emb[k][:])
                prediction = output.max(1, keepdim=True)[1]
                acc = prediction.eq(this_label.view_as(prediction)).sum().item()
                net_acc += acc

                # calculate the loss etc.
                loss = criterion(output, this_label)
                net_loss += loss.item()
        print('log: test loss = {:.2f}, test accuracy = {:.2f}%'.format(net_loss / len(test_examples),
                                                               100.0 * net_acc / len(test_examples)))
        pass

class Another_lstm(nn.Module):

    def __init__(self):
        super(Another_lstm, self).__init__()
        self.half_1 = nn.Sequential(
            nn.LSTM(10, 2, num_layers=2)

        )
        pass

    def forward(self, x):

        pass

    def train_model(self):

        pass

    def test_model(self):

        pass


def main():
    # we receive the addresses as strings and their hit/miss labels as integers
    train_addresses, train_labels = get_dataset(file_name='train.txt', one_hot=False)
    test_addresses, test_labels = get_dataset(file_name='test.txt', one_hot=False)
    print('log: train data size: {} {}'.format(len(train_addresses), train_labels.size()))
    print('log: test data size: {} {}'.format(len(test_addresses), test_labels.size()))

    # next, we need a routine to convert each address into a sequence of vectors that we could feed into an lstm
    # and also we need to make them the same size as well
    def address_to_vector(addr, target_length):
        # step 1. make same size
        while len(addr) < target_length:
            addr = '0'+addr
        # step 2. turn into a matrix, basically target_length*10, target_length for the digits in the address,
        #         and 10 for the 10 possible digits at each spot, so... (and we assume a batch size = 1)
        vec = torch.zeros(target_length, 1, 10) # since there are 10 possible digits!
        for index, digit in enumerate(addr):
            vec[index][0][int(digit)] = 1
        return vec

    # perform a small sanity check
    # print(address_to_vector('6577939', 10))

    n_embeddings = 10
    n_digits = 10
    n_hidden = 16
    n_categories = 2
    model = LSTM_net(input_size=n_digits, embedding_size=n_embeddings, hidden_size=n_hidden, output_size=n_categories)

    # a small dummy test to see if we are in sound stead!
    if False:
        input = address_to_vector('677645', 10)
        hidden = torch.zeros(1, n_hidden)
        output, next_hidden = model(input[0], hidden)
        print(output)

    model.train_model(train_data=(train_addresses, train_labels), test_data=(test_addresses, test_labels),
                      embeddings_function=address_to_vector, epochs=10, learning_rate=0.001, log_after=1000)


if __name__ == '__main__':
    main()










