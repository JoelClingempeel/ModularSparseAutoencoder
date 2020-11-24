# Learn Stripes
This repo contains experiments with different ways of learning stripes with the eventual aim of being used in a neural net based on the prefrontal cortex.
The first step towards realizing this is to build something like an autoencoder where the neurons in the code layer are divided into groups such that only
a limited number of groups may be active at once and train it in such a way that the groups specialize appropriately.


## stripes.py
Main model for store-ignore-recall task.  

* Uses two kinds of sparsity
    -  k-sparsity across an entire layer (ignoring boundaries between stripes)
    -  k-sparsity across stripes ranked by average activation.

## lifetime_sparsity.py
Similar to stripes.py but for sparsity across layers uses lifetime sparsity (computed across a batch).

## independent_stripes.py
Each stripe functions as an independent autoencoder, and only the one with highest average activation may be active.

## boosted_stripes.py
Similar to stripes.py but adjusts sparsity across layers with boosting to encourage different neurons to be active over time.
