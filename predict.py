import pandas as pd
from keras.models import model_from_json
import numpy as np

# Load the Model Architecture
with open('./model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load the Best Model Weights
model.load_weights('model_weights.h5')

# Load the Data for Prediction
new_data = pd.read_csv('./new_data.csv')  # Replace 'path/to/your/new_data.csv' with the relative path to your new data file

# Preprocess the new data: remove commas and convert to float
new_data['Open'] = new_data['Open'].str.replace(',', '').astype(float)
new_data['High'] = new_data['High'].str.replace(',', '').astype(float)
new_data['Low'] = new_data['Low'].str.replace(',', '').astype(float)
new_data['Close'] = new_data['Close'].str.replace(',', '').astype(float)
new_data['Volume'] = new_data['Volume'].str.replace(',', '').astype(float)

X_pred = new_data[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)

# Make Predictions
predictions = model.predict(X_pred)
print("Predictions:")
print(predictions)
