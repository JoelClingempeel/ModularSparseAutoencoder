Experiments with SAY STUFF


## stripes.py
Main model for store-ignore-recall task.  

* Uses two kinds of sparsity
    -  1) k-sparsity across an entire layer (ignoring boundaries between stripes)
    -  2) k-sparsity across stripes ranked by average activation.

## lifetime_sparsity.py
Similar to stripes.py but replaces 1) with lifetime sparsity (computed across a batch).

## independent_stripes.py
Each stripe functions as an independent autoencoder, and only the one with highest average activation may be active.

## boosted_stripes.py
Similar to stripes.py but adjusts 1) with boosting to encourage different neurons to be active over time.
