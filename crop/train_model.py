import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "Crop_recommendation.csv"
df = pd.read_csv(file_path)

# Check for missing values
if df.isnull().sum().sum() > 0:
    df.dropna(inplace=True)

# Select features
features = ['temperature', 'humidity', 'ph', 'rainfall']  # Consider more relevant features
X = df[features]
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  # Tuned hyperparameters
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model
joblib.dump(model, "croNarayangarhp_model.pkl")
print("✅ Model saved as crop_model.pkl")
