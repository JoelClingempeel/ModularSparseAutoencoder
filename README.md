# Modular Sparse Autoencoder
The aim of this project is to experiment with ways of building sparse autoencoders which are modular in the sense that the code layer neurons are divided into clusters (called **stripes** in reference to stripes in the prefrontal cortex) such that only a limited number of clusters may be active at once.

Each experiment uses two kinds of sparsity:
* k-sparsity across an entire layer (ignoring boundaries between stripes).
* k-sparsity across stripes ranked by average activation.

### k-sparsity across an entire layer (ignoring boundaries between stripes):
Controlled by the *layer_sparsity_mode* flag.
* none
* ordinary
    -  The k neurons with highest activations remain active.
* lifetime
    -  Sparsity is computed across a batch to encourage a wider range of neurons to be active.
    -  Reference:  https://arxiv.org/abs/1409.2752
* boosted
    -  Sparsity is enhanced via boosting to make recently active neurons less likely to be active again.
    -  Reference:  https://arxiv.org/abs/1903.11257

### k-sparsity across stripes:
Controlled by the *stripe_sparsity_mode* flag.
* none
* ordinary
    -  The k stripes with highest average activations remain active.
* routing
    -  Each gate is turned on or off as controlled by selecting the top k after applying a linear transformation to the layer before the stripes.
    -  When using this mode, one can set the *routing_l1_regularization* flag to introduce additional (soft) stripe sparsity by regularizing the routing layer.

**Note:**  Tensorboard data is logged in paths of the form

```[log_dir]/[layer_sparsity_mode]/[stripe_sparsity_mode]/[timestamp]/[stripe]```

so that for each digit, different stripes may be shown on the same graph. It is recommended to give tensorboard a specific timestamp by running

```tensorboard --logdir [log_dir]/[layer_sparsity_mode]/[stripe_sparsity_mode]/[timestamp]```

to avoid cluttering with data across runs.
