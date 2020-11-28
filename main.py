import argparse
import datetime
import os

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Net

parser = argparse.ArgumentParser()

# Architecture Flags
parser.add_argument('--intermediate_dim', type=int, default=250)
parser.add_argument('--stripe_dim', type=int, default=20)
parser.add_argument('--num_stripes', type=int, default=15)
parser.add_argument('--num_active_neurons', type=int, default=15)
parser.add_argument('--num_active_stripes', type=int, default=4)
parser.add_argument('--layer_sparsity_mode', type=str, default='none')  # Set to none, ordinary, boosted, or lifetime.
parser.add_argument('--stripe_sparsity_mode', type=str, default='routing')  # Set to none, ordinary, or routing.

# Boosting Flags - Only necessary when mode is set to boosted.
parser.add_argument('--alpha', type=float, default=.8)
parser.add_argument('--beta', type=float, default=1.2)

# Training Flags
parser.add_argument('--lr', type=float, default=.01)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--data_path', type=str, default='data.csv')
parser.add_argument('--log_path', type=str, default='logs')

args = vars(parser.parse_args())


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
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / (batch_size * batch_no)


def log_summary_data(net, writers):
    stripe_stats = net.get_stripe_stats(X_test, Y_test)
    for stripe in range(args['num_stripes']):
        stripe_writer = writers[stripe]
        for digit in range(10):
            stripe_writer.add_scalar(f'digit_{digit}', stripe_stats[digit][stripe], epoch)
        stripe_writer.flush()


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
          args['num_active_stripes'],
          args['layer_sparsity_mode'],
          args['stripe_sparsity_mode'],
          args['alpha'],
          args['beta'])
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),
                      lr=args['lr'],
                      momentum=args['momentum'])
timestamp = str(datetime.datetime.now()).replace(' ', '_')
root_path = os.path.join(args['log_path'],
                         args['layer_sparsity_mode'], 
                         args['stripe_sparsity_mode'],
                         timestamp)
writers = [SummaryWriter(os.path.join(root_path, str(num)))
           for num in range(args['num_stripes'])]

for epoch in range(args['num_epochs']):
    print(f'Epoch number {epoch}')
    loss = train_epoch(net, criterion, optimizer, X_train, batch_size, batch_no)
    print(f'Average Loss: {loss}')
    log_summary_data(net, writers)
