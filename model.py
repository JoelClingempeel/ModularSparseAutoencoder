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
                 layer_sparsity_mode,
                 stripe_sparsity_mode,
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

        if layer_sparsity_mode not in ['none', 'ordinary', 'boosted', 'lifetime']:
            raise ValueError('Layer sparsity mode must be set to none, ordinary, boosted, or lifetime.')
        if stripe_sparsity_mode not in ['none', 'ordinary', 'routing']:
            raise ValueError('Stripe sparsity mode must be set to none, ordinary, or routing.')
        self.layer_sparsity_mode = layer_sparsity_mode
        self.stripe_sparsity_mode = stripe_sparsity_mode

        if layer_sparsity_mode == 'boosted':
            self.alpha = alpha
            self.beta = beta
            self.gamma = int(num_active_neurons / (stripe_dim * num_stripes))
            self.boosted_scores = torch.zeros(stripe_dim * num_stripes, requires_grad=False)

        if stripe_sparsity_mode == 'routing':
            self.routing_layer = nn.Linear(intermediate_dim, num_stripes)

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

    def batch_lifetime_sparsify_layer(self, x):  # Applied to a batch.
        top_neurons = x.mean(0).topk(self.num_active_neurons).indices
        mask = torch.tensor([1 if index in top_neurons else 0
                             for index in range(self.stripe_dim * self.num_stripes)])
        x = mask * x
        return x

    def sparsify_stripes(self, x):
        output = []
        for sample in x:
            avg_values = torch.mean(sample, 1)
            cluster_indices = avg_values.topk(self.num_active_stripes).indices
            mask = torch.tensor([1 if j in cluster_indices else 0
                                 for j in range(self.num_stripes)])
            output.append((sample.transpose(0, 1) * mask).transpose(0, 1))
        return torch.stack(output, dim=0)

    def routing_sparsify_stripes(self, intermediate, stripe_data):
        routing_scores = self.routing_layer(intermediate)
        mask_data = []
        for data in routing_scores:
            cluster_indices = data.topk(self.num_active_stripes).indices
            mask = torch.tensor([1 if j in cluster_indices else 0
                                 for j in range(self.num_stripes)])
            mask_data.append(mask)
        mask = torch.stack(mask_data, dim=0).unsqueeze(2)
        return mask * stripe_data

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
