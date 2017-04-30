# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:03:51 2017

@author: Aditya Gogoi
"""

        # Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


        # Part 2 - Making the ANN!
# Importing the Keras Library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# Improving the ANN
# Using 'Dropout Regularization' to reduce overfitting if needed.
# Dropout decreases the correlation between neurons by disabling them. This decreases overfitting.
from keras.layers import Dropout

# Initialising the ANN as a sequence of layers
# First defining the classifier
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
# 1. output_dim = Number of nodes in the hidden layer. Usually, average of number of input nodes (11) and number of output nodes(1)
# 2. init = Is related to the weights that will be assigned to the synapses. A uniform parameter assigns weights close to 0.
# 3. activation = The activation function being used. 'relu' is the name for rectifier function, which will be used for hidden layers.
# Sigmoid function will be used for Output layer.
# 4. input_dim = Is needed for the first hidden layer as we need to know the number of input nodes that will be connected.
# input_dim will not be required for the next hidden layers.
# We should add the Dropout after each layer of the ANN.
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# We can add a Dropout after every Hidden layer
# 1. p = The ratio of neurons you want to disable. We should start with a lower value (like 0.1) and increase it (by 0.1) if overfitting persists. If we increase it to 1, then we will have underfitting, as no neurons will be enabled. 
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the Output Layer
# Changing output_dim to 1, because outp node is just 1 (for True or False)
# Changing activation to 'sigmoid' as we need a probability value for churn
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# NOTE: User warning about change in Keras 2.0. Change init to kernel_initializer

# Compiling the ANN and optimizing the weights
# 1. optimizer = the method we will use to optimize the weights of our ANN. This will be a form of Gradient Descent called 'adam'.
# 2. loss = Used to calculate the difference between the predicted and actual values.
# We will use a loss function compatible with sigmoid function which is called logarithmic loss. This function in keras is called 'binary_crossentropy'.
# We are using Binary Crossentropy because we have 2 outputs. For Categorical output, we use Softmax function for loss calculation.
# 3. metrics = Theses are the metrics based on which ANN will be optimized. We cacn give a list of metrics, but we will only provide 'accuracy' as the metric.   
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set
# X_train = the independant variable input.
# y_train = the dependant variable input.
# batch_size = Stochastic gradient descent usually tweaks weights after every observation, but we can decide using batch_size parameter.
# nb_epoch = An epoch is when every record of the dataset has been through the NN. The more the number of epochs, the more accurate the predictions become. But we have to optimize through estimation. 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

        # Part 3 - Making Predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Converting probabilities into 1s and 0s, using a threshold of 50%
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy = (Sum of major diagonal elements of confusion matrix)/Total number of observations
# Here we get accuracy of = (1483+224)/2000 = 0.8535
        # Homework Assignment
"""Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000"""

# Creating encoded row for the new observation
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


        # Part 4 - Evaluating, Improving and Tuning the ANN
# First do the data preprocessing (till creation of training and test set)

# Evaluating the ANN
# k-Fold cross validation is present in scikit learn but we have to use keras as well.
# We will use a keras wrapper to assimilate keras with k-fold

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense

# Creating a definition to build the ANN architecture
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Now we will use the Keras Classifier to buid the ANN architecture
# 1. build_fn = the function used to build the ANN architecture, in this case the function we just developed
# 2. batch_size and 3. nb_epoch = same as the reasons given above.
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)

# Now using k-fold cross validation to train our classifier
# 1. estimator = The object used for fitting i.e. the ANN.
# 2. X = the independant variables i.e. X_train
# 3. y_train = the dependant variable i.e. y_train
# 4. cv = the number of folds to split the training data into
# 5. n_jobs = the numper 2of CPUs to perform the calculations. With larger number of folds come more calculations, hence more CPUs required. -1 = all CPUS
# The accuracies given by each iteration will be stored in the 'accuracies' array.

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

# Tuning the parameters
# Certain parameters like predictions and accuracies keep on changing.
# But there are also Hyper-Parameters which do not change like number of epochs, iterations, etc.
# We can tune these parameters to improve accuracy.
# We will use the GridSearchCV module
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
# In order to change hyper parameters within a function, then include that parameter in that function call and include it in the parameters dictionary.
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)

# Now we define a dictionary which consists of hyper parameters and the values we want them to have.
parameters = {'batch_size' : [25, 32],
              'nb_epoch' : [100, 500],
              'optimizer' : ['adam','rmsprop']} 

# rmsprop is a stochastic gradient descent optimizer used mostly in RNN
# Now we will implement GridSearchCV object
# 1. estimator = the object we want to optimize i.e. the ANN classifier
# 2. param_grid = the dictionary of parameters that we wanna incoporate in the grid_search object
# 3. scoring = the aspect baed on which we will evaluate the model, in this case it is accuracy.
# 4. cv = the number of folds we want to split the dataset into.
# This Grid search is an extension of crass-validation from sklearn.
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

# Fitting the model with the Training dataset and storing it in the same object name.
grid_search = grid_search.fit(X_train, y_train)

# Storing the best parameters and score among all the diffirent evaluations
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# This actually works
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
 





