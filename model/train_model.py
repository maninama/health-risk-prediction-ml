import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Dataset load karo
data = pd.read_csv("../dataset/health_data.csv")

# Features (X) aur Target (y) split
X = data.drop("risk", axis=1)
y = data["risk"]

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML Pipeline (Scaling + Model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000
    ))
])

# Model training
pipeline.fit(X_train, y_train)

# Model evaluation
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Trained model save karo
with open("model.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Model training completed & model.pkl saved")
