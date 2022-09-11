import numpy as np


class Activation_ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self , dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0 ] = 0

    def predictions(self, outputs):
        return outputs

        
class Activation_Softmax:
    def forward(self , inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis = 1,keepdims = True))
        probabilities = exp_values / np.sum(exp_values,axis = 1,keepdims = True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
            enumerate ( zip (self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape( - 1 , 1 )
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
            single_dvalues)
        
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Activation_Sigmoid:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1
    

class Activation_Linear:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs


class Activation_Softmax_Loss_CategoricalCrossentropy:

    def backward(self, dvalues, y_true):
        samples = len (dvalues)
        if len (y_true.shape) == 2 :
            y_true = np.argmax(y_true, axis = 1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[ range (samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
