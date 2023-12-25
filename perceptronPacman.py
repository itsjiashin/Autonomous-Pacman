# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation for apprenticeship learning
import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import matplotlib.pyplot as plt
# from data_classifier_new import convertToArray
import math
import numpy as np

PRINT = True


class SingleLayerPerceptronPacman():

    def __init__(self, num_weights=5, num_iterations=20, learning_rate=1):

        # weight initialization
        # model parameters initialization

        # initialise all features to be 1s
        self.weights = np.ones(num_weights)

        self.max_iterations = num_iterations
        self.learning_rate = learning_rate


    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and compute
        the dot product of the weights of your perceptron with the values of features.

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        Then the result of this computation should be passed through your activation function

        For example if x=feature_vector, and ReLU is the activation function
        this function should compute ReLU(x dot weights)
        """

        "*** YOUR CODE HERE ***"
        return self.activation(np.dot(feature_vector, self.weights))

    def activation(self, x):
        """
        Implement your chosen activation function here.
        """

        "*** YOUR CODE HERE ***"
        #Sigmoid activation function is used
        return 1 / (1 + np.exp(-x))

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the perceptron.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable.

        The data should be a 2D numpy array where the each row is a feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """

        "*** YOUR CODE HERE ***"
        #A basic accuracy testing is performed
        correct_pred_num = 0
        for i in range(len(data)):
            pred = self.predict(data[i])
            if (pred >= 0.5 and labels[i] == 1) or (pred < 0.5 and labels[i] == 0):
                correct_pred_num += 1
        return correct_pred_num/len(data)


    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This function should take training and validation data sets and train the perceptron

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """

        "*** YOUR CODE HERE ***"
        training_loss = []
        validation_loss = []
        for _ in range(self.max_iterations):
            #Feed forward, we first get all the predictions for the train and validation dataset
            train_predictions = np.array([self.predict(feature_vector) for feature_vector in trainingData])
            validation_predictions = np.array([self.predict(feature_vector) for feature_vector in validationData])

            #The gradients will always be np.zeros(5), because within our weights there are 1 bias + 4 features
            new_weights = np.zeros(5)
            #For every element in the gradient numpy array, use the gradient descent algorithm for MSE taught in the lectures to update
            for i in range(5):
                for j in range(len(train_predictions)):
                    new_weights[i] += (train_predictions[j] - trainingLabels[j]) * train_predictions[j]*(1 - train_predictions[j]) * trainingData[j][i]
            #Every element within the gradient numpy array is multiplied by 2/N, where N is the length of the training data
            new_weights = new_weights*(2/len(trainingData))
            #Update the weights based on the formula given in the slides
            self.weights = self.weights - self.learning_rate*new_weights

            #https://www.geeksforgeeks.org/python-mean-squared-error/
            training_loss.append(np.square(np.subtract(trainingLabels, train_predictions)).mean())
            validation_loss.append(np.square(np.subtract(validationLabels, validation_predictions)).mean())
        #     print("Validation Accuracy for 750 iterations are: " + str(self.evaluate(validationData, validationLabels)))
        # print("Training loss: " + str(training_loss[-1]))
        # print("Validation loss: " + str(validation_loss[-1]))
        #Plot out the training and validation loss over the number of iterations 
        plt.figure(figsize=(10, 5))
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Iteration/Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()



