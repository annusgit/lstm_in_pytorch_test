

from __future__ import print_function
from __future__ import division
import os
import shutil
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms


class Network(nn.Module):

    def __init__(self):
        """
            This is another way of defining a model in a smaller code
        """
        super(Network, self).__init__()
        self.half_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.half_2 = nn.Sequential(
            nn.Linear(100352/64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.half_1(x)
        x = x.view(-1, 100352/64)
        x = self.half_2(x)
        return x

    def num_flat_features(self, x):
        dimensions = x.size()[1:] # get all dimensions except the batch size!
        # now just multiply them all...
        num_features = 1
        for dim in dimensions:
            num_features *= dim
        return num_features

    def train_model(self, train_data, test_data, epochs, save_dir='models', load_from_saved=False,
                    model_to_load_from='', log_after=100, lr=1e-3):
        self.train()
        if load_from_saved:
            checkpoint = torch.load(model_to_load_from)
            self.load_state_dict(checkpoint)
            print('log: loaded from saved model {}. Running test now...'.format(model_to_load_from))
            self.test_model(test_data=test_data)
        optimizer = optim.SGD(params=self.parameters(), lr=lr, momentum=0.5)
        criterion = nn.CrossEntropyLoss()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for e in range(1, epochs+1):
            for idx, (batch_examples, batch_labels) in enumerate(train_data):
                # zero out the gradients
                optimizer.zero_grad()
                output = self.forward(batch_examples)
                # print(output.size(), batch_labels.size())
                loss = criterion(output, batch_labels)
                # backprop
                loss.backward()
                optimizer.step()
                if idx % log_after == 0:
                    print("log: epoch# {} ({}/{}), batch loss = {:0.2f}".format(e, idx*len(batch_examples),
                                                                                len(train_data.dataset), loss.item()))
            model_name = os.path.join(save_dir, 'model-{}.ckpt'.format(e))
            print('log: saving {} now...'.format(model_name))
            # we can save whatever we want...
            torch.save(self.state_dict(), model_name)
            self.test_model(test_data)
        pass

    def test_model(self, test_data):
        print('\n\tlog: testing now...\n')
        self.eval()
        accumulated_loss = 0.0
        net_acc = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_examples, batch_labels in test_data:
                output = self.forward(batch_examples)
                loss = criterion(output, batch_labels).item()
                predictions = output.max(1, keepdim=True)[1]
                acc = predictions.eq(batch_labels.view_as(predictions)).sum().item()
                net_acc += acc
                accumulated_loss += loss
            print("log: net loss = {:0.2f}, net accuracy = ({}/{}) = {:0.2f}%".format(
                accumulated_loss / len(test_data.dataset),
                net_acc, len(test_data.dataset),
                100.0 * net_acc / len(test_data.dataset)))
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

    net = Network()
    net.train_model(train_loader, test_loader, save_dir='models', load_from_saved=True,
                    model_to_load_from='models/model-1.ckpt', epochs=10, log_after=100, lr=0.01)


if __name__ == '__main__':
    main()




