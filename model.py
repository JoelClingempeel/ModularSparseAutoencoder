import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# For a batch of vectors.
def k_mask(x, k):
    thresholds = x.topk(k).values[:, -1]
    return x >= (thresholds.unsqueeze(1) * torch.ones(x.shape))

# For a batch of striped vectors.
def stripe_k_mask(x, k):
    return k_mask(torch.mean(x, 2), k) > torch.zeros(x.shape[:-1])

class Net(nn.Module):
    def __init__(self,
                 intermediate_dim,
                 stripe_dim,
                 num_stripes,
                 num_active_neurons,
                 num_active_stripes,
                 layer_sparsity_mode,
                 stripe_sparsity_mode,
                 distort_prob,
                 alpha,
                 beta,
                 active_stripes_per_batch,
                 device):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(784, intermediate_dim)
        self.layer2 = nn.Linear(intermediate_dim, stripe_dim * num_stripes)
        self.layer3 = nn.Linear(stripe_dim * num_stripes, intermediate_dim)
        self.layer4 = nn.Linear(intermediate_dim, 784)

        self.stripe_dim = stripe_dim
        self.num_stripes = num_stripes
        self.num_active_neurons = num_active_neurons
        self.num_active_stripes = num_active_stripes

        if layer_sparsity_mode not in ['none', 'ordinary', 'boosted', 'lifetime']:
            raise ValueError('Layer sparsity mode must be set to none, ordinary, boosted, or lifetime.')
        if stripe_sparsity_mode not in ['none', 'ordinary', 'routing']:
            raise ValueError('Stripe sparsity mode must be set to none, ordinary, or routing.')
        self.layer_sparsity_mode = layer_sparsity_mode
        self.stripe_sparsity_mode = stripe_sparsity_mode

        self.distort_prob = distort_prob

        if layer_sparsity_mode == 'boosted':
            self.alpha = alpha
            self.beta = beta
            self.gamma = int(num_active_neurons / (stripe_dim * num_stripes))
            self.boosted_scores = torch.zeros(stripe_dim * num_stripes, requires_grad=False).to(device)

        if stripe_sparsity_mode == 'routing':
            self.routing_layer = nn.Linear(intermediate_dim, num_stripes)

        self.active_stripes_per_batch = active_stripes_per_batch

        self.device = device

    def _distort_mask(self, mask):
        rand_mask = torch.rand(mask.shape) > 1 - self.distort_prob
        return torch.logical_xor(mask, rand_mask)

    def _boosts(self):
        return torch.exp(self.beta * (self.gamma - self.boosted_scores)).to(self.device)

    def batch_sparsify_layer(self, x):
        return k_mask(x, self.num_active_neurons) * x

    def batch_boosted_sparsify_layer(self, x):
        # Calculate mask with boosting, then update boost scores, and then apply mask.
        boosted_x = self._boosts() * x
        thresholds = boosted_x.topk(self.num_active_neurons).values[:, -1]
        mask = boosted_x >= thresholds.unsqueeze(1) * torch.ones(boosted_x.shape)
        with torch.no_grad():
            self.boosted_scores *= (1 - self.alpha)
            self.boosted_scores += self.alpha * mask.sum(0)
        return mask * x

    def batch_lifetime_sparsify_layer(self, x):  # Applied to a batch.
        avg_values = x.mean(0)
        threshold = avg_values.topk(self.num_active_neurons).values[-1]
        mask = avg_values >= threshold * torch.ones(avg_values.shape)
        return mask * x

    def sparsify_stripes(self, x):
        return self._distort_mask(stripe_k_mask(x, self.num_active_stripes)).unsqueeze(2) * x

    def lifetime_sparsify_stripes(self, x):
        num_active = math.ceil(self.active_stripes_per_batch * len(x))
        stripe_avg_values = torch.mean(x, 2).transpose(0, 1)
        thresholds = stripe_avg_values.topk(num_active).values[:, -1]
        mask = stripe_avg_values >= thresholds.unsqueeze(1) * torch.ones(stripe_avg_values.shape)
        mask = self._distort_mask(mask)
        return mask.transpose(0, 1).unsqueeze(2) * x

    def routing_sparsify_stripes(self, intermediate, stripe_data):
        routing_scores = self.routing_layer(intermediate)
        mask = self._distort_mask(k_mask(routing_scores, self.num_active_stripes))
        return mask.unsqueeze(2) * stripe_data

    def encode(self, x):
        x = F.relu(self.layer1(x))
        stripe_data = F.relu(self.layer2(x))

        if self.layer_sparsity_mode == 'ordinary':
            stripe_data = self.batch_sparsify_layer(stripe_data)
        elif self.layer_sparsity_mode == 'boosted':
            stripe_data = self.batch_boosted_sparsify_layer(stripe_data)
        elif self.layer_sparsity_mode == 'lifetime':
            stripe_data = self.batch_lifetime_sparsify_layer(stripe_data)

        stripe_data = stripe_data.reshape(-1, self.num_stripes, self.stripe_dim)
        if self.stripe_sparsity_mode == 'ordinary':
            stripe_data = self.sparsify_stripes(stripe_data)
        elif self.stripe_sparsity_mode == 'routing':
            stripe_data = self.routing_sparsify_stripes(x, stripe_data)

        if self.active_stripes_per_batch < 1:
            stripe_data = self.lifetime_sparsify_stripes(stripe_data)
        return stripe_data

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
        zero_stripe = torch.zeros(self.stripe_dim).to(self.device)
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
            x_var = torch.FloatTensor(X[k: k + 1])
            x_var = x_var.to(self.device) 
            stripes = self.get_active_stripes(x_var)
            for stripe in stripes:
                activations[digit][stripe] += 1
        return activations

    def get_routing_scores(self, x):
        x = F.relu(self.layer1(x))
        return self.routing_layer(x)

    def get_average_activations(self, X, Y, device='cpu'):
        running_activations = {}
        running_counts = {}
        for digit in range(10):
            running_activations[str(digit)] = torch.zeros(self.num_stripes, self.stripe_dim).to(self.device)
            running_counts[str(digit)] = 0

        with torch.no_grad():
            for datum, label in zip(X, Y):
                x_var = torch.FloatTensor(datum).unsqueeze(0)
                x_var = x_var.to(device)
                digit = str(label.item())
                running_activations[digit] += self.encode(x_var).squeeze(0)
                running_counts[digit] += 1

        return torch.stack([running_activations[str(digit)] / running_counts[str(digit)]
                            for digit in range(10)],
                           dim=0).to(self.device)

