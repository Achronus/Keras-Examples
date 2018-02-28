# Recurrent Neural Networks (RNN)

#----------------------
# Data Preprocessing
#----------------------
# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # Only want the open column but want it as a numpy array

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# Create data structure with 60 timesteps and 1 output
X_train = [] # Will contain the 60 previews timesteps (the inputs)
y_train = [] # Will contain the outputs
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0]) # 0 specifies the column index
    y_train.append(training_set_scaled[i, 0])
    
# Convert X_train & y_train into numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape data structure into 3D to fit into RNN
# RNN takes an input tuple as: (batchsize, timesteps, input_dim)
# (X_train.shape[0] (1,198) = batch_size; X_train.shape[1] (60) = timesteps; 1 = input_dim)
reshaped_X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#-------------------
# Building the RNN
#-------------------
# Import Keras Libraries
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Initalize the RNN
regressor = Sequential()

# Adding the first LSTM layer and dropout
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))

# Adding the second LSTM layer and dropout
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the third LSTM layer and dropout
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the fourth LSTM layer and dropout
regressor.add(LSTM(units=150))
regressor.add(Dropout(rate=0.2))

# Output Layer
regressor.add(Dense(units=1))

# Comping the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting to the training set
regressor.fit(X_train, y_train, batch_size=32, epochs=100)


#------------------------
# Making the predictions 
#------------------------
# Get the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Concat both of the datasets
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

# Get the inputs - used to predict the accuracy against the real stock prices
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

# Reshape the inputs into 1 column
inputs = inputs.reshape(-1, 1)

# Scale the inputs to match the training set
# This has already been fitted to the training set so we only need to transform it
inputs = sc.transform(inputs)

# Create the data structure for the test set
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
    
# Convert to a numpy array
X_test = np.array(X_test)
    
# Reshape the data structure into a 3D shape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the prediction stock price of 2017
predicted_stock_price = regressor.predict(X_test)

# Convert predictions back to readable format - no longer feature scaled
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#------------------------
# Visualise the results
#------------------------
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()