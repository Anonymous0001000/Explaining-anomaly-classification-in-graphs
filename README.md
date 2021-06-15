__This repository provides an implementation of the method presented in the paper__

*Explaining anomaly classification in graphs*

The paper is currently under review at ICDM 2021.  

# Dependencies:
Currently works with
* tensorflow==1.14.0
* keras==2.2.4
* numpy>=1.16.1
* scikit-learn==0.20.2
* networkx>=2.4
* h5py==2.9.0
* typing
* random

# Usage
After installing the dependencies, you can run the code by typing:

'python -m code.main --save save_file.h5 --epochs 500 --anomaly_param 3 --loops'

This will create a file 'save_file.h5' to save the weights of the GNN and train it for 500 epochs.

'--anomaly_param' sets the type of anomaly.

'--loops' add self loops to the graph

# Pre-trained model
You can load pre-trained weights for each type of anomaly.
For example, use '--save A_0.h5' for '--anomaly_param 0'.
Consider fine-tuning with '--epochs number_of_epochs' if you use pre-trained weights. 
