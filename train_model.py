import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("Cancer_Data.csv")

# Drop unnecessary column
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# Convert labels
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

# Features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model + feature names
pickle.dump((model, X.columns.tolist()), open('model.pkl', 'wb'))

print("✅ Model created successfully!")