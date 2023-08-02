import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np

# Step 1: Load and Preprocess Data
data = pd.read_csv('data.csv')

# Replace commas and hyphens from 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume' columns
data['Open'] = data['Open'].str.replace(',', '').str.replace('-', '')
data['High'] = data['High'].str.replace(',', '').str.replace('-', '')
data['Low'] = data['Low'].str.replace(',', '').str.replace('-', '')
data['Close'] = data['Close'].str.replace(',', '').str.replace('-', '')
data['Adj Close'] = data['Adj Close'].str.replace(',', '').str.replace('-', '')
data['Volume'] = data['Volume'].str.replace(',', '').str.replace('-', '')

# Convert the columns to float and handle missing values
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
data['High'] = pd.to_numeric(data['High'], errors='coerce')
data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Adj Close'] = pd.to_numeric(data['Adj Close'], errors='coerce')
data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

# Drop rows with missing values
data.dropna(inplace=True)

X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
y = data['Adj Close'].values.astype(np.float32)

# Step 2: Build the Model
model = Sequential()
model.add(Dense(64, input_shape=(5,), activation='relu'))
model.add(Dense(1))

# Step 3: Compile the Model
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

# Step 4: Train the Model
model_checkpoint = ModelCheckpoint('model_weights.h5', save_best_only=True)  # Change filepath to include the filename
model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[model_checkpoint])

# Step 5: Save the Model Architecture
with open('./model.json', 'w') as json_file:
    json_file.write(model.to_json())

print("Training completed. Model and weights saved.")
