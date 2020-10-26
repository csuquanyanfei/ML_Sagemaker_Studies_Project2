import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
            # number of hidden nodes in each layer (512)
      
        # linear layer (input_dim -> hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # linear layer (n_hidden -> output_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)
        self.input_dim=input_dim
        self.sig = nn.Sigmoid()
        # define all layers, here
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
         # flatten image input
        #x = x.view(-1, self.input_dim)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc2(x)
        return self.sig(x)