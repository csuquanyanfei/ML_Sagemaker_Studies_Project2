# torch imports
import torch.nn.functional as F
import torch.nn as nn
import torch


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
         # linear layer (input_features-> hidden_dim)
        self.fc1 = nn.Linear(input_features, hidden_dim,bias=True)
        # linear layer (n_hidden -> output_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,bias=True)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim,bias=True)
        self.input_features, self.hidden_dim, self.output_dim = input_features, hidden_dim, output_dim
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
       # self.dropout = nn.Dropout(0.2)
    
        #self.sig = nn.Sigmoid()

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        nn.init.constant_(self.fc1.bias.data,1)
        nn.init.constant_(self.fc2.bias.data,1)
        nn.init.constant_(self.fc3.bias.data,1)
        
        x = torch.sigmoid(self.fc1(x))      # activation function for hidden layer
        x = torch.sigmoid(self.fc2(x))      # hiddenlayer 2
        x = torch.sigmoid(self.fc3(x))      # linear output
        # add dropout layer
        #x = self.dropout(x)
        # add output layer
        #x = self.fc2(x)
        #x = self.dropout(x)
        #x = self.fc3(x)
        return x
    