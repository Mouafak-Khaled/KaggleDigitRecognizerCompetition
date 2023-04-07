# KaggleDigitRecognizerCompetition
This project aims to provide a solution to the Kaggle Digit Recognizer Challenge.

## Competition Overview:
* Number of Teams: 1459
* Competitors: 1461
* My Leaderboard: 506

# The Architecture of DigitRecognizer:

##Residual Block:
* A 2D convolution Layer
  * Kernel Size: 3 x 3
  * Padding: 1
* Batch Normalization Layer
* ReLU Activataion
* A 2D convolution Layer
  * Kernel Size: 3 x 3
  * Padding: 1
* Batch Normalization Layer
* ReLU Activataion
* Skip Connection

##1. First Layer:

* A 2D convolution layer:
  * Kernel size of 3 x 3.
  * 2 Filters
* Batch normalization
* ReLU Activation

##2.  Second Layer:

* A 2D convolution layer:
  * Kernel size of 3 x 3.
  * 4 Filters
* Batch normalization
* Max Pooling Layer:
  * Kernel Size: 2 x 2
  * Stride: 2 x 2
* ReLU Activation
* Droput2d Layer

## 3. Third Layer:

* A 2D convolution layer:
  * Kernel size of 3 x 3.
  * 8 Filters
* Batch normalization
* ReLU Activation
* Droput2d Layer

## 4. Fourth Layer:

* A 2D convolution layer:
  * Kernel size of 3 x 3.
  * 16 Filters
* Batch normalization
* Max Pooling Layer:
  * Kernel Size: 2 x 2
  * Stride: 2 x 2
* ReLU Activation
* Droput2d Layer

## 5. Fifth Layer:

* A 2D convolution layer:
  * Kernel size of 3 x 3.
  * 10 Filters
* Batch normalization
* ReLU Activation
* Droput2d Layer

## A ResidualBlock is applied after each layer in the convolutional neural network.
## An Adaptive Average Pooling layer is used as a last layer instead of a fully-connected layer.
  

# Training:

## 1. Optimizer:
* AdamW.
  * Learning rate of 2e-2.
  * Weight decay of 1e-3.
  * Amsgrad.

## 2. Criterion (Loss function):
* Cross entropy loss.

## 3. Epochs:
* The model was trained for 30 epochs

## 4. Learning rate scheduler:
* ReduceLROnPlateau.
  * Gamma: 0.1
  * patience: 5


# Evaluation:
* Accuracy:

| Train | Validation | Test |
|-------|-------------|------|
| 99.97%| 99.11%      | 99.04%|

* Precision, recall, and f1-score:

| classes | precision | recall | f1-score | support |
|---------|-----------|--------|----------|---------|
| Zero       | 0.99     | 0.99   | 0.99     | 980    |
| One       | 0.99     | 0.99   | 0.99     | 1135    |
| Two     | 0.99      | 1.0   | 0.99     | 1032     |
| Three       | 0.98      | 1.0   | 0.99     | 1010     |
| Four     | 0.99       | 0.99   | 0.99     | 982     |
| Five       | 0.99      | 0.98   | 0.98     | 892    |
| Six     | 0.99       | 0.99   | 0.99     | 958     |
| Seven       | 0.99       | 0.99   | 0.99     | 1028     |
| Eight     | 0.99      | 0.99   | 0.99     | 974     |
| Nine     | 0.99      | 0.99   | 0.99     | 1009     |



# Dataset:
The dataset used for training this CNN is MNIST dataset.
The dataset was downloaded using torchvision.datasets module in PyTorch.

