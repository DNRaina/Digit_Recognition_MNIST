# Digit_Recognition_MNIST
MNIST Digit Recognizer

This is a simple digit recognition project using the MNIST dataset. The model is built using Convolutional Neural Networks (CNN) and achieves 97% accuracy on the test set.
Overview

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). This project builds a CNN model to classify these digits.
Features

    Input: 28x28 grayscale images
    Model: Convolutional Neural Network (CNN)
    Accuracy: 97% on the test set
    Libraries used:
        TensorFlow/Keras
        NumPy
        Matplotlib (for visualization)

Model Architecture

The CNN model consists of:

    Conv2D layers: Extracting features from images
    MaxPooling layers: Reducing dimensionality
    Fully Connected layers: For classification

Requirements

To run this project, you'll need the following dependencies:

bash

pip install tensorflow numpy matplotlib

How to Run

    Clone the repository:

    bash

git clone <repository-url>
cd mnist-digit-recognizer

Run the training script:

bash

    python train.py

    After training, the model will evaluate the test data and display the accuracy.

Dataset

The MNIST dataset is automatically downloaded and loaded using Keras:

python

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

Results

The model achieves 97% accuracy on the test dataset after training for a few epochs. Example predictions can be visualized using Matplotlib.