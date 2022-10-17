
# Mini Deep Learning library

## About

Barebone neural network library, written with python and numpy only. 

The idea behind this project is to study innerworkings of deep learning algorithms and different neural netowork structures. 

The library has following components:
 - [Layers](nn/layer.py) ( Dense, Dropout, BatchNorm )
 - [Activations](nn/activation.py) ( ReLu, Softmax, Sigmoid, Linear )
 - [Loss](nn/loss.py) ( MSE, MAE, Categorical Cross entropy, Binary Cross entropy )
 - [Optimizers](nn/optimizer.py) ( SGD, Adagrad, RMSprop, Adam )
 - [Accuracy metrics](nn/accuracy.py) ( Regression, Categorical )
 - [Model](nn/model.py) ( compile, build, evaluate, fit, log )

## Installation
    $ git clone https://github.com/kturaevv/nn_mini
    $ cd nn_mini
    $ pip install -r requirements.txt
    
## Training

The basic model can be initialized with 2 lines of code, following predefined architecture. At this moment there is only 1 "template" for model compilation -> `"basic"`:

 - Layer - `Dense`
 - Activation - `ReLu`
 - Output activation - `Softmax`
 - Loss - `Categorical Cross Entropy`
 - Optimizer - `Adam`

To create a Deep Neural Network with 1000 input dimensions, 2 hidden layers of size 128 and output layer with 10 output nodes,  do this:

```python
model = nn.model.Model('basic', [1000, 128, 128, 10])
model.train(*args, **kwargs)
```

However more formal way of structuring a model would be following Keras like API:

```python
# Init model
model_manual = nn.model.Model()

# Add layers, sequentally
model_manual.add(nn.layer.Dense(n_inputs, n_outputs, activation=nn.activation.ReLU()))
        ... # Setting as many layers as needed
model_manual.add(nn.layer.Dense(n_inputs, n_outputs, activation=nn.activation.Softmax()))

# Setting components of choice
model_manual.set(
    loss = nn.loss.CategoricalCrossentropy(),
    accuracy = nn.accuracy.Categorical(),
    optimizer = nn.optimizer.Adam(learning_rate=learning_rate, decay=5e-5)
)

# Train and test the model
model_manual.train(*args, **kwargs)

```

*Note: first 2 input arguments of each layer correspond to number of weights each neuron should have in a layer. This is done to make the layer setting more explicit and implicitly refer to the way tensors are calculated ( A dot product between 2D matrices. )*

It is also possible to mix mentioned methods. Using template like structure for most similar layers and then editing or adding new elements where needed. 
This is possible because the model contains all layers in a `model.layers` and all trainable layers in `model.trainable_layers`.  Pseudocode:

```py
# Setting general structure
model = Model('basic', [*network_structure_and_size])

# Structure editing
model.layers[index].activation = new_activation_function
    ... # other layer editing statements

# This will still work, practically overriding default components
model.set(optimizer=nn.optimizer.SGD) # will set SGD instead of Adam

model.train(*args, **kwargs)
```

Each component is loosely coupled, thus, it is also possible to build model piece by piece, following PyTorch design conventions.

## Benchmark

To see whether the model works, the model have been [compared](https://github.com/kturaevv/nn_mini/blob/main/examples/fashion_mnist/basic_comparison.ipynb
) with PyTorch and Tensorflow Keras, on fashion_MNIST dataset:
