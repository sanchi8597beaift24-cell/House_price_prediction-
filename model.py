import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("house_data.csv")

# Input and output
X = data[['Area', 'Bedrooms', 'Age']]
y = data['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved!")