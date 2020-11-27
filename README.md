# Learn Stripes
This repo contains experiments with different ways of learning stripes with the eventual aim of being used in a neural net based on the prefrontal cortex.
The first step towards realizing this is to build something like an autoencoder where the neurons in the code layer are divided into groups such that only
a limited number of groups may be active at once and train it in such a way that the groups specialize appropriately.


Each experiment uses two kinds of sparsity:
* k-sparsity across an entire layer (ignoring boundaries between stripes).
* k-sparsity across stripes ranked by average activation.

The former has three versions as determined by the *mode* flag.
* ordinary
    -  The k neurons with highest activations remain active.
* lifetime
    -  Sparsity is computed across a batch to encourage a wider range of neurons to be active.
    -  Reference:  https://arxiv.org/abs/1409.2752
* boosted
    -  Sparsity is enhanced via boosting to make recently active neurons less likely to be active again.
    -  Reference:  https://arxiv.org/abs/1903.11257

**Note:**  Tensorboard data is logged in paths of the form *[log_dir]/[mode]/[timestamp]/[stripe]* so that for each digit, different stripes may be shown on the same graph. It is recommended to give tensorboard a specific timestamp (i.e. by running *tensorboard --logdir [log_dir]/[mode]/[timestamp]*) to avoid cluttering with data across runs.
