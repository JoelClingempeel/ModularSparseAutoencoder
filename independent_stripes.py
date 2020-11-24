import argparse

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

parser = argparse.ArgumentParser()

# Architecture Flags
parser.add_argument('--intermediate_dim', type=int, default=250)
parser.add_argument('--stripe_dim', type=int, default=20)
parser.add_argument('--num_stripes', type=int, default=15)
# Training Flags
parser.add_argument('--lr', type=float, default=.01)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--num_epochs', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--data_path', type=str, default='data.csv')

args = vars(parser.parse_args())


class Stripe(nn.Module):
    def __init__(self, intermediate_dim, hidden_dim):
        super(Stripe, self).__init__()
        self.layer1 = nn.Linear(784, intermediate_dim)
        self.layer2 = nn.Linear(intermediate_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, intermediate_dim)
        self.layer4 = nn.Linear(intermediate_dim, 784)

        self.hidden_dim = hidden_dim

    def encode(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x

    def decode(self, x):
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class PFC:
    def __init__(self, num_stripes, stripe_intermediate_dim, stripe_hidden_dim):
        self.stripes = [Stripe(stripe_intermediate_dim, stripe_hidden_dim)
                        for _ in range(num_stripes)]

    def select_stripe(self, x):
        return np.argmax([stripe.encode(x).mean().item()
                          for stripe in self.stripes])

    def parameters(self):
        parameters = []
        for stripe in self.stripes:
            parameters += list(stripe.parameters())
        return parameters

    def __call__(self, x):
        index = self.select_stripe(x)
        return self.stripes[index](x), index


def train_epoch(net, criterion, optimizer, data, labels, batch_size, batch_no):
    activation_data = {}
    for k in range(10):
        activation_data[k] = {}

    data = shuffle(data)
    total_loss = 0
    for i in range(batch_no):
        batch_loss = 0
        optimizer.zero_grad()

        for j in range(i * batch_size, (i + 1) * batch_size):
            x_var = torch.FloatTensor(data[j: j + 1])
            xpred_var, active_stripe = net(x_var)
            batch_loss += criterion(xpred_var, x_var)
            activation_data[labels[j]][active_stripe] = activation_data[labels[j]].get(active_stripe, 0) + 1

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
    return total_loss / (batch_size * batch_no), activation_data

data = pd.read_csv(args['data_path']).values
Y = data[:, :1].transpose()[0]
X = data[:, 1:] / 255
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
batch_size = args['batch_size']
batch_no = len(X_train) // batch_size

net = PFC(10, 200, 100)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),
                      lr=args['lr'],
                      momentum=args['momentum'])

for epoch in range(args['num_epochs']):
    print(f"Epoch number {epoch}")
    loss, activation_data = train_epoch(net, criterion, optimizer, X_train, Y_train, batch_size, batch_no)
    print(f'Average Loss: {loss}')
    for digit, stats in activation_data.items():
        print(f'{digit}:\t{stats}')

