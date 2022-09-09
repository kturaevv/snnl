import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l1 = 0, bias_regularizer_l2 = 0):
        
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self , dvalues):
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

# Dropout
class Layer_Dropout :
    def __init__ ( self , rate ):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    def forward ( self , inputs ):
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial( 1 , self.rate,
        size = inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward ( self , dvalues ):
        self.dinputs = dvalues * self.binary_mask


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self , dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0 ] = 0

        
class Activation_Softmax:
    def forward(self , inputs):
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


class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.ext(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        
            
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    
    def regularization_loss(self, layer):
        regularization_loss = 0
        
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        
        if layer.weight_regularizer_l2 > 0:
            regularization_loss = layer.weight_regularizer_l2 * np.sum(layer.weights*layer.weights)
        
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases*layer.biases)
        
        return regularization_loss

            
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

        
class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__ (self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

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


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = \
                self.learning_rate * (1./(1.+self.decay*self.iterations))
    
    def update_params(self, layer):
        if self.momentum:
            # if layer does not contain momentum arrays -> create
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else: # Vanilla SGD updates
            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates = - self.current_learning_rate * layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates
            
    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad :
    def __init__ ( self , learning_rate = 1. , decay = 0. , epsilon = 1e-7 ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            ( 1. / ( 1. + self.decay * self.iterations))

    def update_params ( self , layer ):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr (layer, 'weight_cache' ):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += - self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += - self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1
    
class Optimizer_RMSprop:
    
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    # Call once before any parameter updates
    def pre_update_params (self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            ( 1. / ( 1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
        ( 1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + \
        ( 1 - self.rho) * layer.dbiases ** 2
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += - self.current_learning_rate * \
            layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += - self.current_learning_rate * \
            layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:
    
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            ( 1. / ( 1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)
        
        # Update momentum with current gradient
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases
        
        # Get corrected momentum ?
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        
        # Update cache with square current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            ( 1 - self.beta_2) * layer.dbiases ** 2
        
        # Get corrected cache?
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1 ))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1 ))
        
        # Vanilla SGD + normalization with square rooted cache
        layer.weights += - self.current_learning_rate * \
            weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += - self.current_learning_rate * \
            bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1