import numpy as np
import pandas as pd

class Node:
    def __init__(self, n_weights: int) -> None:
        # Identity of an object, i.e., a Node
        self.node_id = str(id(self))[7:]
        
        # Create random weights that match the number of inputs
        # TODO optimize later
        # 1 bias per node and n weights for each corresponding input per node
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=n_weights) 
        self.bias = 0 # np.random.uniform(low=0.0, high=10.0, size=None)
        
    def __str__(self) -> str:
        return f"Node {self.node_id}|{len(self.weights)}"
    
    def out(self, input_):
        return np.sum(self.weights*input_) + self.bias


class NeuralNetwork():
    def __init__(self, n_inputs, hidden_layers_struct, n_outputs) -> None:
        self.inputs = n_inputs
        self.outputs = n_outputs
        
        self.network_structure = [n_inputs] + hidden_layers_struct
        
        self.hidden_layers = self.__construct_hidden_layers__()
    
    def show_structure(self) -> str:
        print("Info: ", len(self.hidden_layers), " hidden layers")
        
        for indx, layer in enumerate(self.hidden_layers):
            print(f"Hidden layer {indx + 1} has {len(layer)} nodes: \n", 
                  np.array(
                      [f"Node {n.node_id}|{len(n.weights)} weights" for n in layer]
                  ).reshape(-1,1), '\n')
    
    def __construct_hidden_layers__(self):
        weights = [i for i in self.network_structure]
        matrix = []
        
        for layer in self.network_structure[1:len(self.network_structure)]:
            n_weights = weights.pop(0)
            matrix.append([Node(n_weights) for n in range(layer)])
        return np.array(matrix, dtype=object)
    
    def forward_propagate(self, input_, layer=0):
        if layer == len(self.hidden_layers):
            return input_
        else:
            layer_output = []
            for node in self.hidden_layers[layer]:
                layer_output.append(node.out(input_))
            print(f"Layer {layer} input: \n", input_)
            print(f"Layer {layer} output: \n", layer_output, "\n")
            return self.forward_propagate(layer_output, layer+1)
    
    def _input_layer_manual(self, *args):
        nodes = list() #empty list

        num = int(input("How many input nodes do you want: ")) #user tells the range(optional)

        #iterating till user's range is reached
        while num: 
            value = int(input("Enter a value between 0 and 1: "))

            if value in [0,1]:#asking for input of 1 value 
                nodes.append(value)#adding that value to the list
                num -= 1
            else:
                print("Please provide value between 0 and 1: ")

        print("Input nodes", nodes)
        return nodes


if __name__ == '__main__':

    inp = [1,2,3,4]
    node = Node(inp)
    print(node.do())

