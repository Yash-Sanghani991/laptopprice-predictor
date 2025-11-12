import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load and clean data
df = pd.read_csv("real_estate.csv")  # Replace with actual file
df = df.dropna()

# Simple preprocessing
df['location'] = df['location'].str.lower()
dummies = pd.get_dummies(df['location'])
df = pd.concat([df[['total_sqft', 'bhk', 'price']], dummies], axis=1)

X = df.drop("price", axis=1)
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
data = {
    "model": model,
    "columns": list(X.columns)
}
with open("model.pkl", "wb") as f:
    pickle.dump(data, f)
