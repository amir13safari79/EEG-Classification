# EEG-Classification
### The primary goal of this project, which consists of two phases, is to classify EEG signals from [BCI2003](https://www.bbci.de/competition/iii/) data to classify right and left-hand movement. I did this project for the EE-SUT computational intelligence course in Fall 2021.

<br>

# Classification Procedure and results
## phase1:
### In this phase, first, I extracted some Statistical and frequency features from training data such as mean skewness, entropy, median frequency, frequency-bands energy, etc. 
Then I tried to find the 40 best features with one-dimensional fisher criterion and classify signals with neural networks with Variable parameters like hidden layer size, training function, etc.

Final accuracy in this phase with 5-fold validation was 67.09% and 67.41% using MLP and RBF NNs, respectively.

<br>

## phase2:
### In the 2nd phase, using the optimtool Matlab toolbox and genetic algorithms with multi-dimensional fisher criterion as a fitness function, I try to find the best 15 features from 40 features found in phase1 to reduce the dimension of features.
Final accuracy in this phase with 5-fold validation was 66.46% and 66.77% using MLP and RBF NNs, respectively.

<br>

### for more detail please study report of this project.

