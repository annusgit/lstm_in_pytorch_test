
"""
    Learn torch by example
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch import optim
from torch.tensor import Tensor
import torch.nn.functional as F
from torchvision import datasets, transforms


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(1, 10, kernel_size=5, padding=1)
        self.conv_2 = nn.Conv2d(10, 20, kernel_size=5, padding=1)
        self.conv_2_dropout = nn.Dropout2d()
        self.conv_3 = nn.Conv2d(20, 32, kernel_size=5, padding=1)
        # define linear layers, like y = A*x+b
        self.fc_1 = nn.Linear(32, 64)
        self.fc_2 = nn.Linear(64, 10)
        pass

    def forward(self, x):
        """
            Formally perform the feed forward through the network
        :param x: in features
        :return: final output, a 1x10 vector of predictions
        """
        x = F.max_pool2d(F.relu(self.conv_1(x)), kernel_size=(2, 2))
        x = self.conv_2_dropout(F.max_pool2d(F.relu(self.conv_2(x)), kernel_size=(2, 2)))
        x = F.max_pool2d(F.relu(self.conv_3(x)), kernel_size=(2, 2))
        x = x.view(-1, self.num_flat_features(x)) # basically flatten the result to input into the fully conn. layer
        x = F.relu(self.fc_1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        dimensions = x.size()[1:] # get all dimensions except the batch size!
        # now just multiply them all...
        num_features = 1
        for dim in dimensions:
            num_features *= dim
        return num_features

    def train_model(self, train_data, test_data, learn_rate=1e-3, epochs=10, log_after=100):
        self.train()
        optimizer = optim.SGD(self.parameters(), lr=learn_rate, momentum=0.5)
        for e in range(1, epochs+1):
            for idx, (batch_examples, batch_labels) in enumerate(train_data):
                # print(batch_labels[)
                output = self.forward(batch_examples)
                # print(output.size(), batch_labels.size())
                loss = F.nll_loss(output, batch_labels)
                # backprop, but first zero out the gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % log_after == 0:
                    print("log: epoch# {} ({}/{}), batch loss = {:0.2f}".format(e, idx*len(batch_examples),
                                                                                len(train_data.dataset), loss.item()))
            self.test_model(test_data)
        pass

    def test_model(self, data):
        print('\n\tlog: testing now...\n')
        self.eval()
        accumulated_loss = 0.0
        net_acc = 0
        with torch.no_grad():
            for batch_examples, batch_labels in data:
                output = self.forward(batch_examples)
                loss = F.nll_loss(output, batch_labels, size_average=False).item()
                predictions = output.max(1, keepdim=True)[1]
                acc = predictions.eq(batch_labels.view_as(predictions)).sum().item()
                net_acc += acc
                accumulated_loss += loss
            print("log: net loss = {:0.2f}, net accuracy = ({}/{}) = {:0.2f}%".format(accumulated_loss/len(data.dataset),
                                                                                      net_acc, len(data.dataset),
                                                                                      100.0*net_acc/len(data.dataset)))
        pass


def main():
    train_batch_size, test_batch_size = 64, 64
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True)

    model = Net()
    model.train_model(train_data=train_loader, test_data=test_loader, learn_rate=0.01, epochs=10, log_after=100)

    pass


if __name__ == '__main__':
    main()






