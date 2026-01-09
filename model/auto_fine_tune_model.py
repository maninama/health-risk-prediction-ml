import pandas as pd
import pickle
import mysql.connector

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# MySQL se data fetch
def fetch_mysql_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="health_risk_db"
    )

    query = """
        SELECT age, gender, fever, bp, sugar, oxygen, predicted_risk
        FROM health_records
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # column rename (dataset ke match ke liye)
    df.rename(columns={"predicted_risk": "risk"}, inplace=True)
    return df


# Old CSV dataset load
old_data = pd.read_csv("../dataset/health_data.csv")

# MySQL data load
mysql_data = fetch_mysql_data()

# Merge old + new data
full_data = pd.concat([old_data, mysql_data], ignore_index=True)

print("Total records after merge:", len(full_data))

# X & y split
X = full_data.drop("risk", axis=1)
y = full_data["risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fine-tuned pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=2.0,
        class_weight="balanced",
        max_iter=3000
    ))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Auto Fine-Tuned Accuracy:", acc)

# Save updated model
with open("model.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Auto fine-tuned model saved as model.pkl")
