# UCRNet
UCRNet: Underwater color image restoration via a polarization-guided convolutional neural network

This is our dataset and code for the paper "UCRNet: Underwater color image restoration via a polarization-guided convolutional neural network"


Please cite this work when using this code and dataset.


################################
How to train？

Configure the libraries required by the project and use the main.py file to start training.
Pay attention to the import path of the dataset and hyperparameter configuration in the project!
The folder named rawtrain is the training dataset and the folder named evalF is the test dataset.

How to test?
If you change the value of the data variable to 2022.08.02, the program will start using the trained model for testing.
The model is trained on two graphics cards in parallel. Directly using checkpoint for testing requires two or more graphics cards.
Pay attention to the import path of the checkpoint and hyperparameter configuration in the project!
