import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv("breast_cancer_data.csv")

# Clean data
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Split features & target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]
feature_names = X.columns.tolist()


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate (optional)
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save feature names (order matters!)
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("Model and scaler saved successfully.")
