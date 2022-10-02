from abc import ABC, abstractmethod
import functools
from platform import architecture
import numpy as np
from typing import Type


from nn.activation import Activation


class Layer(ABC):

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Layer_Input(Layer):

    def __init__(self, inputs=None):
        self.output = inputs

    def forward(self, inputs):
        self.output = inputs

    def backward(self):
        pass    


class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, activation: Type[Activation],
                 weight_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l1 = 0, bias_regularizer_l2 = 0,
                 ):
        
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        # Set previous and next layers, for clarity
        self.prev = None
        self.next = None

        # Activation function as a decorator
        self.activation = activation

    def __activation__(function):
        """ Activation function is rather a wrapper than another layer."""

        @functools.wraps(function)
        def forward(self, inputs, *args, **kwargs):
            # Activation for forward propagation passes sequentially
            # layer.forward -> layer.activation -> layer.output
            function(self, inputs)
            return self.activation.forward(inputs = self.output)
        
        @functools.wraps(function)
        def backward(self, dvalues):
            # Activation for bacward propagation passes in reversed order
            # layer.activation -> layer.backward -> layer.output
            self.activation.backward(dvalues)
            return function(self, self.activation.dinputs)
        
        if function.__name__ == 'forward':
            return forward
        elif function.__name__ == 'backward':
            return backward

    @__activation__
    def forward(self, inputs, training=None):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    @__activation__
    def backward(self, dvalues, training=None):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0 , keepdims = True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0 :
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0 ] = - 1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0 :
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0 :
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0 ] = - 1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0 :
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


# Dropout
class Dropout(Layer):
    def __init__ (self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial( 1 , self.rate, size = inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class BatchNormalization(Layer):
    
    def __init__(self, momentum=0.99, eps=0.01, r_mean=None, r_var=None):
        self.momentum = momentum
        self.eps = eps
        self.running_mean = r_mean
        self.running_var = r_var

        # Initialize the parameters, params to be learned
        self.weights  = np.ones(self.input_shape) # Gamma as weight, to make layer trainable
        self.bias = np.zeros(self.input_shape) # Beta as bias
        
    def forward(self, inputs, training=True):
            
        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = np.mean(inputs, axis=0)
            self.running_var = np.var(inputs, axis=0)

        if training:
            mean = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        # Normalization
        # x - mean / Standard deviation

        self.X_centered = inputs - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)

        X_norm = self.X_centered * self.stddev_inv
        self.output = self.weights * X_norm + self.biases

    def backward(self, dvalues):
        
        # Save parameters used during the forward pass
        gamma = self.weights

        X_norm = self.X_centered * self.stddev_inv
        self.dweights = np.sum(dvalues * X_norm, axis=0)
        self.dbiases = np.sum(dvalues, axis=0)

        self.gamma = self.gamma_opt.update(self.gamma, self.dweights)
        self.beta = self.beta_opt.update(self.beta, self.dbiases)

        batch_size = dvalues.shape[0]

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)
        dvalues = (1 / batch_size) * gamma * self.stddev_inv * (
            batch_size * dvalues
            - np.sum(dvalues, axis=0)
            - self.X_centered * self.stddev_inv**2 * np.sum(dvalues * self.X_centered, axis=0)
            )

        self.dinputs = dvalues

    def output_shape(self):
        return self.input_shape