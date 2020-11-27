import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,
                 intermediate_dim,
                 stripe_dim,
                 num_stripes,
                 num_active_neurons,
                 num_active_stripes,
                 mode,
                 alpha,
                 beta):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(784, intermediate_dim)
        self.layer2 = nn.Linear(intermediate_dim, stripe_dim * num_stripes)
        self.layer3 = nn.Linear(stripe_dim * num_stripes, intermediate_dim)
        self.layer4 = nn.Linear(intermediate_dim, 784)

        self.stripe_dim = stripe_dim
        self.num_stripes = num_stripes
        self.num_active_neurons = num_active_neurons
        self.num_active_stripes = num_active_stripes
        if mode not in ['ordinary', 'boosted', 'lifetime']:
            raise ValueError('Mode must be set to boosted or lifetime.')
        self.mode = mode

        if mode == 'boosted':
            self.alpha = alpha
            self.beta = beta
            self.gamma = int(num_active_neurons / (stripe_dim * num_stripes))
            self.boosted_scores = torch.zeros(stripe_dim * num_stripes, requires_grad=False)

    def _boosts(self):
        return torch.exp(self.beta * (self.gamma - self.boosted_scores))

    def sparsify_layer(self, x):  # Applied to an individual image.
        indices = x.topk(self.num_active_neurons).indices
        mask = torch.tensor([1 if j in indices else 0
                             for j in range(len(x))])
        return x * mask

    def batch_sparsify_layer(self, x):
        return torch.stack([self.sparsify_layer(data) for data in x], dim=0)

    def batch_boosted_sparsify_layer(self, x):
        # Calculate masks with boosting, then update boost scores, and then apply masks.
        with torch.no_grad():
            masks = torch.stack([self.sparsify_layer(self._boosts() * datum)
                                 for datum in x],
                                dim=0)
            self.boosted_scores *= (1 - self.alpha)
            self.boosted_scores += self.alpha * masks.sum(0)
        return masks * x

    def batch_lifetime_sparsify(self, x):  # Applied to a batch.
        top_neurons = x.mean(0).topk(self.num_active_neurons).indices
        mask = torch.tensor([1 if index in top_neurons else 0
                             for index in range(self.stripe_dim * self.num_stripes)])
        x = mask * x
        return x

    def sparsify_stripes(self, x):
        avg_values = torch.mean(x, 1)
        cluster_indices = avg_values.topk(self.num_active_stripes).indices
        mask = torch.tensor([1 if j in cluster_indices else 0
                             for j in range(len(avg_values))])
        return (x.transpose(0, 1) * mask).transpose(0, 1)

    def encode(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        if self.mode == 'ordinary':
            x = self.batch_sparsify_layer(x)
        elif self.mode == 'boosted':
            x = self.batch_boosted_sparsify_layer(x)
        elif self.mode == 'lifetime':
            x = self.batch_lifetime_sparsify(x)
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
            for j in range(self.num_stripes):
                activations[i][j] = 0

        for k in range(len(Y)):
            digit = Y[k]
            stripes = self.get_active_stripes(torch.FloatTensor(X[k: k + 1]))
            for stripe in stripes:
                activations[digit][stripe] += 1
        return activations
