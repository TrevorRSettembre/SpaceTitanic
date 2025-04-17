# SpaceTitanic

# Requirments 
    pip install
    torch nightly for cuda 12.8
    torchvision
    scikit-learn
    pandas
    numpy
    matplotlib
    tqdm
    torch-lr-finder

# Data
    located in Main project folder /data
    contains the test and training csv

# Run Instructions
    The code defaults to these paramaters listed below so change based on system hardware
    epochs = 150 Complete pass through the entire dataset
    kfolds = 5 the model trains on 4 folds and validates on the 5th
    batch_size = 64 samples the model looks at before updating its weights.
    num_workers = 8 number of CPU threads for loading data in parallel
    patience = 5 used for early stopping terminates run if model does not improve after patience runs