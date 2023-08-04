import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

# Load the data from CSV file
data = pd.read_csv('data.csv')

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date in ascending order
data.sort_values(by='Date', ascending=True, inplace=True)

# Extract the 'Low' column as the target variable (y)
target_column = 'Low'
y = data[target_column].values

# Drop unnecessary columns and 'Low' column from the features
features = data.drop(columns=['Date', 'Open', 'High', 'Close', 'AdjClose', 'Volume']) # Remove 'Low' column

# Convert features and target to numpy arrays
X = features.values

# Normalize the features to scale them between 0 and 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a sequential model
model = Sequential()

# Add a dense layer with 64 neurons and 'relu' activation function
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))

# Add another dense layer with 32 neurons and 'relu' activation function
model.add(Dense(32, activation='relu'))

# Add another dense layer with 16 neurons and 'relu' activation function
model.add(Dense(16, activation='relu'))

# Add the output layer with 1 neuron (since it's a regression problem)
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the training data
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) as a measure of performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Let's assume the data for 3 months later (November 2023)
# Note: Update this data with the actual data for 3 months later
future_data = np.array([[6820.00]])

# Normalize the future_data using the same scaler
future_data_normalized = scaler.transform(future_data)

# Make a prediction for the buying price
predicted_low = model.predict(future_data_normalized)[0][0]
print(f'Predicted buying price for 3 months later: {predicted_low}')
