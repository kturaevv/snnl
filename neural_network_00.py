import numpy as np

from pandas import DataFrame


class Node:
    def __init__(self) -> None:
        self.node_id = str(id(self))[7:] # Identity of an object, i.e., a Node
        
        self.weight = np.random.uniform(low=0.0, high=1.0, size=None)
        self.bias = np.random.uniform(low=0.0, high=10.0, size=None)
        
        self.node_val = 0

    def __str__(self) -> str:
        return f"Node {self.node_id}"

    def activation_function(self, *args):
        for in_value in args:
            self.node_val += self.__step_activation(in_value)
    
    def __step_activation(self, arg):
        pass

    def __sigmoid_activation(self, arg):
        pass

    def __relU_activation(self, arg):
        pass


class NerualNetwork():
    def __init__(self, n_nodes, n_layers) -> None:
        self.hidden_layers = self._construct_hidden_layers(n_nodes, n_layers)
    
    def __str__(self) -> str:
        return str(DataFrame(self.hidden_layers))

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

    def _input_layer_from_source(self, *args):
        return [args]

    def _construct_hidden_layers(self, n_nodes, n_layers):
        matrix = [[Node() for _ in range(n_nodes)] for _ in range(n_layers)]
        return matrix

    def _output_layer(self):
        pass


if __name__ == '__main__':
    nn = NerualNetwork(5,5)
    print(nn)

    while True:
        x = input("Continue? (y/n) ")
        if x == 'n':
            break
        
        # nodes = nn._input_layer_manual()
