import argparse

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
parser.add_argument('--num_active_neurons', type=int, default=15)
parser.add_argument('--num_active_stripes', type=int, default=4)
# Training Flags
parser.add_argument('--lr', type=float, default=.01)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--num_epochs', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--data_path', type=str, default='data.csv')

args = vars(parser.parse_args())


class Net(nn.Module):
    def __init__(self, intermediate_dim, stripe_dim, num_stripes, num_active_neurons, num_active_stripes):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(784, intermediate_dim)
        self.layer2 = nn.Linear(intermediate_dim, stripe_dim * num_stripes)
        self.layer3 = nn.Linear(stripe_dim * num_stripes, intermediate_dim)
        self.layer4 = nn.Linear(intermediate_dim, 784)

        self.stripe_dim = stripe_dim
        self.num_stripes = num_stripes
        self.num_active_neurons = num_active_neurons
        self.num_active_stripes = num_active_stripes

    def sparsify_layer(self, x):
        indices = x.topk(self.num_active_neurons).indices
        mask = torch.tensor([1 if j in indices else 0
                             for j in range(len(x))])
        return x * mask

    def sparsify_stripes(self, x):
        avg_values = torch.mean(x, 1)
        cluster_indices = avg_values.topk(self.num_active_stripes).indices
        mask = torch.tensor([1 if j in cluster_indices else 0
                             for j in range(len(avg_values))])
        return (x.transpose(0, 1) * mask).transpose(0, 1)

    def encode(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.stack([self.sparsify_layer(data) for data in x], dim=0)
        x = x.reshape(-1, self.num_stripes, self.stripe_dim)
        return torch.stack([self.sparsify_stripes(data) for data in x], dim=0)

    def decode(self, x):
        x = x.reshape(-1, self.num_stripes * self.stripe_dim)
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def get_active_stripes(self, x):
        code = self.encode(x).squeeze(0)
        zero_stripe = torch.zeros(self.stripe_dim)
        return [j for j, stripe in enumerate(code)
                if not torch.all(torch.eq(stripe, zero_stripe))]

    def get_stripe_stats(self, X, Y):
        activations = {}
        for i in range(10):
            activations[i] = {}
        for j in range(len(Y)):
            stripes = str(self.get_active_stripes(torch.FloatTensor(X[j: j + 1])))
            activations[Y[j]][stripes] = activations[Y[j]].get(stripes, 0) + 1
        return activations


def train_epoch(net, criterion, optimizer, data, batch_size, batch_no):
    data = shuffle(data)
    total_loss = 0
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = torch.FloatTensor(data[start:end])

        optimizer.zero_grad()
        xpred_var = net(x_var)
        loss = criterion(xpred_var, x_var)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / (batch_size * batch_no)


def display(stats, cutoff):
    for digit in range(10):
        print(f'Digit {digit}')
        for pattern, count in stats[digit].items():
            if count >= cutoff:
                print(f'{pattern}\t\t\t{count}')
        print('\n')


data = pd.read_csv(args['data_path']).values
Y = data[:, :1].transpose()[0]
X = data[:, 1:] / 255
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
batch_size = args['batch_size']
batch_no = len(X_train) // batch_size

net = Net(args['intermediate_dim'],
          args['stripe_dim'],
          args['num_stripes'],
          args['num_active_neurons'],
          args['num_active_stripes'])
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),
                      lr=args['lr'],
                      momentum=args['momentum'])

for epoch in range(args['num_epochs']):
    print(f"Epoch number {epoch}")
    loss = train_epoch(net, criterion, optimizer, X_train, batch_size, batch_no)
    print(f'Average Loss: {loss}')
    stripe_stats = net.get_stripe_stats(X_test, Y_test)
    display(stripe_stats, 4)
