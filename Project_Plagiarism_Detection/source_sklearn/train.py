from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib
from skorch import NeuralNetRegressor
from model import BinaryClassifier
from sklearn.model_selection import GridSearchCV

#from sklearn.svm import LinearSVC
## TODO: Import any additional libraries you need to define a model

#Begin Yanfei's first try#
#from sklearn.svm import LinearSVC
#End Yanfei's first try#

#Begin Yanfei's second try#
from sklearn.linear_model import LogisticRegression
#Begin Yanfei's second try#
# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--random_state', type=int, default=0, metavar='N',
                           help='int, RandomState instance, default=0')
    parser.add_argument('--solver', type=str, default='lbfgs', metavar='S',
                         help='Possible values: {newton-cg, lbfgs, liblinear, sag, saga}, default is lbfgs')
    parser.add_argument('--multi_class', type=str, default='ovr', metavar='S',
                        help='Possible values: {auto, ovr, multinomial}, default is ovr')
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    

    ## TODO: Define a model 
    
    #Begin Yanfei's first try#
    #model=LinearSVC()
    #End Yanfei's first try#
    net = NeuralNetRegressor(BinaryClassifier(3,20,1)
                         , max_epochs=100
                         , lr=0.001
                         , verbose=1)
    
    #Begin Yanfei's second try#
    model = LogisticRegression(random_state=args.random_state, solver=args.solver, multi_class=args.multi_class)
    #End Yanfei's first try#
    params = {
    'lr': [0.001,0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'max_epochs': list(range(500,5500, 500))
      }

    ## TODO: Train the model
   # model.fit(train_x,train_y)
    #model = GridSearchCV(net, params, refit=False, scoring='r2', verbose=1, cv=10)

    model.fit(train_x, train_y)
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))