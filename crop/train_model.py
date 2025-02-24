import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "Crop_recommendation.csv"  # Ensure this file is present in the same directory
df = pd.read_csv(file_path)

# Select features (Only temperature & humidity)
X = df[['temperature', 'humidity']]
y = df['label']  # Crop label

# Split dataset for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save trained model
joblib.dump(model, "crop_model.pkl")
print("✅ Model saved as crop_model.pkl")
