import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('combined_unique_output.csv')
df.columns = df.columns.str.strip()

# Encode 'Soil Type' (categorical)
soil_encoder = LabelEncoder()
df['Soil Type'] = soil_encoder.fit_transform(df['Soil Type'])

# Encode target column 'Crop Type'
crop_encoder = LabelEncoder()
df['Crop Type'] = crop_encoder.fit_transform(df['Crop Type'])

# Drop 'Fertilizer Name' since it's not used for crop prediction
df = df.drop(['Fertilizer Name'], axis=1)

# Features and target
X = df.drop('Crop Type', axis=1)
y = df['Crop Type']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders
with open('Model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('crop_encoder.pkl', 'wb') as f:
    pickle.dump(crop_encoder, f)

with open('soil_encoder.pkl', 'wb') as f:
    pickle.dump(soil_encoder, f)

print("✅ Crop prediction model and encoders saved.")
