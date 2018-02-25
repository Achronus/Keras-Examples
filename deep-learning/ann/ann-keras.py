#----------------------
# Data Preprocessing
#----------------------

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encode the Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X1 = LabelEncoder()
LabelEncoder_X2 = LabelEncoder()

# Categorize the categorical data
X[:, 1] = LabelEncoder_X1.fit_transform(X[:, 1])
X[:, 2] = LabelEncoder_X2.fit_transform(X[:, 2])

# Create the dummy variables
OneHotEncoder = OneHotEncoder(categorical_features=[1])
X = OneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:] # Remove first dummy variable to avoid dummy variable trap

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#----------------------
# Build the ANN
#----------------------

# Importing the libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer, epochs, loss, batch_size, custom_units):
    # Initalise the model
    classifier = Sequential()
    # Add first layer and hidden layer
    classifier.add(Dense(units=custom_units, kernel_initializer='random_uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(rate=0.1))
    # Add second layer
    classifier.add(Dense(units=custom_units, kernel_initializer='random_uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    # Output layer
    classifier.add(Dense(units=1, kernel_initializer='random_uniform', activation='sigmoid'))
    # Compile the ANN
    classifier.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # Fit the ANN to the training set
    classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    return classifier

# Wrap the classifier
classifier = KerasClassifier(build_fn=build_classifier)

# Create the parameters dictionary
parameters = { 
              {'optimizer' : ['adamax'], 'epochs' : [10, 100], 'loss' : ['mean_squared_error', 'mean_absolute_error'], 'batch_size' : [32, 64], 'custom_units' : [32, 64]},
              {'optimizer': ['adam'], 'epochs': [10], 'loss:': ['binary_crossentropy'], 'batch_size': [32], 'custom_units': [32]}
             }

# Create the Grid Search and apply K-Fold Cross Validation
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)

# Identify optimal parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


#-----------------------------------------------------
# Using best parameters with K-Fold Cross Validation
#-----------------------------------------------------
# This is for any future training purposes
from sklearn.model_selection import cross_val_score

# Build the classifier
def build_best_classifier():
    # Initalise the model
    classifier = Sequential()
    # Add first layer and hidden layer
    classifier.add(Dense(units=32, kernel_initializer='random_uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(rate=0.1))
    # Add second layer
    classifier.add(Dense(units=32, kernel_initializer='random_uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    # Output layer
    classifier.add(Dense(units=1, kernel_initializer='random_uniform', activation='sigmoid'))
    # Compile the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Wrap the classifier and fit it to training set
best_classifier = KerasClassifier(build_fn=build_best_classifier, batch_size=32, epochs=10)
best_accuracies = cross_val_score(estimater=best_classifier, X_train, y_train, cv=10)

# Predict the mean and variance
mean = best_accuracies.mean()
variance = best_accuracies.std()


#--------------------------
# Single Predictions
#--------------------------
# Details for the observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60,000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50,000
"""

# Create single row observation
single_observation = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])

# Put the prediction on same scale
single_observation = sc.transform(single_observation)

# Make the prediction
new_prediction = classifier.predict(new_prediction)
new_prediction = (new_prediction > 0.5) # True or false

#--------------------------
# Making predictions
#--------------------------
# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)