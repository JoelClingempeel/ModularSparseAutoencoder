import argparse
import datetime
import os

import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

from model import Net
from train import train

parser = argparse.ArgumentParser()

# Architecture Flags
parser.add_argument('--intermediate_dim', type=int, default=250)
parser.add_argument('--stripe_dim', type=int, default=5)
parser.add_argument('--num_stripes', type=int, default=30)
parser.add_argument('--num_active_neurons', type=int, default=15)
parser.add_argument('--num_active_stripes', type=int, default=5)
parser.add_argument('--layer_sparsity_mode', type=str, default='none')  # Set to none, ordinary, boosted, or lifetime.
parser.add_argument('--stripe_sparsity_mode', type=str, default='routing')  # Set to none, ordinary, or routing.

# Boosting Flags - Only necessary when layer_sparsity_mode is set to boosted.
parser.add_argument('--alpha', type=float, default=.8)
parser.add_argument('--beta', type=float, default=1.2)

# Routing Flags - Only necessary when stripe_sparsity_mode is set to routing.
parser.add_argument('--routing_l1_regularization', type=float, default=0.)
parser.add_argument('--log_average_routing_scores', type=bool, default=True)

# Lifetime Stripe Flag - Only necessary when stripe_sparsity_mode is set to lifetime.
# Within a stripe, this specifies the proportion of samples that may activate the stripe.
parser.add_argument('--active_stripes_per_batch', type=float, default=1.)

# Training Flags
parser.add_argument('--lr', type=float, default=.01)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--data_path', type=str, default='data.csv')
parser.add_argument('--log_path', type=str, default='logs')
parser.add_argument('--log_class_specific_losses', type=bool, default=False)

args = vars(parser.parse_args())


def main(args):
    data = pd.read_csv(args['data_path']).values
    Y = data[:, :1].transpose()[0]
    X = data[:, 1:] / 255
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    num_stripes = args['num_stripes']
    num_epochs = args['num_epochs']
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
              args['beta'],
              args['active_stripes_per_batch'])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=args['lr'],
                          momentum=args['momentum'])

    timestamp = str(datetime.datetime.now()).replace(' ', '_')
    root_path = os.path.join(args['log_path'],
                             args['layer_sparsity_mode'],
                             args['stripe_sparsity_mode'],
                             timestamp)
    print(f'Logging results to path:  {root_path}')

    routing_l1_regularization = (args['routing_l1_regularization'] if args['stripe_sparsity_mode'] == 'routing' else 0)
    log_class_specific_losses = args['log_class_specific_losses']
    should_log_average_routing_scores = (
                args['stripe_sparsity_mode'] == 'routing' and args['log_average_routing_scores'])

    train(net,
          criterion,
          optimizer,
          root_path,
          X_train,
          X_test,
          Y_test,
          num_stripes,
          num_epochs,
          batch_size,
          batch_no,
          routing_l1_regularization,
          log_class_specific_losses,
          should_log_average_routing_scores)


if __name__ == '__main__':
    main(args)
