# De novo design for antibacterial and anticancer peptides via a deep conditional generative model

A deep learning-based conditional generative model for the generation of peptide sequences with the desired antibacterial and anticancer activity

## Dependencies
To install python dependencies, run: `pip install -r requirements.txt`

## Running
`python main.py --batch_size [BATCH_SIZE] --G_units [GRU Unit Size] --emb_dim [Embedding Dim] --G_lr [Learning Rate] --G_iter [Max Num Epoch]`

- --batch_size: Batch size
- --G_units: Number of GRU units. 
    For two-stacked GRU layers with 24 unit sizes: --G_units 24,24 
    For a single GRU layer with 24 unit sizes: --G_units 24
- --emb_dim: Dimension of embedding layer
- --G_lr: Learning rate
- --G_iter: Maximum epoch for model training

## Trained model weights
Saved model weights are located in "./w_and_l/{subpath}".
{subpath} is automatically provided by input parameters.

## Model performance check
The trained model performance can be checked by tensorboard.
`tensorboard --logdir ./w_and_l/`
Detailed tensorboard can be found in https://www.tensorflow.org/tensorboard.
