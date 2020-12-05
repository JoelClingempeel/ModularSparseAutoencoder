# Modular Sparse Autoencoder
The aim of this project is to experiment with ways of building sparse autoencoders which are modular in the sense that the code layer neurons are divided into clusters (called **stripes** in reference to stripes in the prefrontal cortex) such that only a limited number of clusters may be active at once.

Each experiment uses three kinds of sparsity:
* k-sparsity across an entire layer (ignoring boundaries between stripes).
* k-sparsity across stripes ranked by average activation.
* Lifetime stripe sparsity.

### k-sparsity across an entire layer (ignoring boundaries between stripes):
Controlled by the `layer_sparsity_mode` flag.
* `none`
* `ordinary`
    -  The k neurons with highest activations remain active.
* `lifetime`
    -  Sparsity is computed across a batch to encourage a wider range of neurons to be active.
    -  Reference:  https://arxiv.org/abs/1409.2752
* `boosted`
    -  Sparsity is enhanced via boosting to make recently active neurons less likely to be active again.
    -  Reference:  https://arxiv.org/abs/1903.11257

### k-sparsity across stripes:
Controlled by the `stripe_sparsity_mode` flag.
* `none`
* `ordinary`
    -  The k stripes with highest average activations remain active.
* `routing`
    -  Each gate is turned on or off as controlled by selecting the top k after applying a linear transformation to the layer before the stripes.
    -  When using this mode, one can set the `routing_l1_regularization` flag to introduce additional (soft) stripe sparsity by regularizing the routing layer.

### Lifetime stripe sparsity
Controlled by the `active_stripes_per_batch` flag.
* With this flag, a given stripe may be only active for a fixed number of samples **per batch**. The goal is to vary what stripes are active across different samples.
    - This is applied **after** k-sparsity mechanisms across stripes.

### Logging
Tensorboard data for a run is logged in a path of the form:

```[log_dir]/[layer_sparsity_mode]/[stripe_sparsity_mode]/[timestamp]```

To view, run:

```tensorboard --logdir [log_dir]/[layer_sparsity_mode]/[stripe_sparsity_mode]/[timestamp]```

### Automated Hyperparameter Tuning
Modify the `hyperparameters_config.json` so that for each flag it lists the different possibilies you wish to test, and then run `tune_parameters.py`.
(Alternatively, you can use a different config file if you change the path in the `hyperparameters_config` flag in `tune_parameters.py`.)

### Analysis / Visualization
Use Analyze.ipynb in a Jupyter Notebook, and remember that for all experiments under consideration you *must* set the `log_experiment_flags` flag. 
