import numpy as np


            
class Loss:

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss    

        return data_loss, self.regularization_loss()
    
    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.trainable_layers:

            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            
            if layer.weight_regularizer_l2 > 0:
                regularization_loss = layer.weight_regularizer_l2 * np.sum(layer.weights*layer.weights)
            
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases*layer.biases)
            
        return regularization_loss


class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        sample_loss = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_loss

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2*(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
    

class Loss_CategoricalCrossentropy(Loss):
    # Loss function applied to one hot encoded expected values
    # i.e. if at the output only 1 node should be fired up
    
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        
        # Clippling means putting array in a range from -> to
        # constituting unsatisfying values with those in range
        # i.e. ~0.0000001 < y_pred < ~0.9999999
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7 )
        
        if len(y_true.shape) == 1:
            # if format of expected values is [~y1, ... ~yn]
            # an array of expected indices, i.e. categorical data,
            # get all rows and values only at index y_true
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            # if format of expected values is a matrix, i.e. one hot encoded
            
            # 0 1 2 index 
            # -----
            # 1 0 0
            # 0 1 0
            # 0 0 1
            
            # element wise product of y_pred and y_true 
            # and sum -> by horizontal axis will result in 1D array
            # where each element represents pred value or 0
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        
        negative_log_likelihoods = - np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len (dvalues)
        labels = len (dvalues[ 0 ])
        if len (y_true.shape) == 1 :
            y_true = np.eye(labels)[y_true]
        self.dinputs = - y_true / dvalues
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples
