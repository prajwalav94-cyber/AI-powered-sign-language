import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the data
df = pd.read_csv("gesture_data.csv")

# Last column is label
label_col = df.columns[-1]

X = df.drop(label_col, axis=1)
y = df[label_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=500)
model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)
print("Accuracy:", acc)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl!")

