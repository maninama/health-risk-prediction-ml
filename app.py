from flask import Flask, render_template, request
import pickle
from db import get_connection
import os
import csv
import pandas as pd

app = Flask(__name__, template_folder="templates")

# ---------- PATH ----------
CSV_PATH = "dataset/health_data.csv"

# ---------- Load ML model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ---------- Home ----------
@app.route("/")
def home():
    return render_template("form.html")

# ---------- SUBMIT ROUTE (ALIAS) ----------
@app.route("/submit", methods=["POST"])
def submit():
    return predict()

# ---------- Predict + Save ----------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ---- 1. Read form data ----
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        fever = int(request.form["fever"])
        bp = int(request.form["bp"])
        sugar = int(request.form["sugar"])
        oxygen = int(request.form["oxygen"])

        print("Form data received:", age, gender, fever, bp, sugar, oxygen)

        # ---- 2. ML Prediction ----
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "fever": fever,
            "bp": bp,
            "sugar": sugar,
            "oxygen": oxygen
        }])

        prediction = int(model.predict(input_data)[0])
        confidence = float(round(max(model.predict_proba(input_data)[0]) * 100, 2))

        print("Prediction:", prediction)
        print("Confidence:", confidence)

        risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        risk_class_map = {0: "low", 1: "medium", 2: "high"}

        # ---- 3. DB TRANSACTION ----
        print("---- DB TRANSACTION START ----")

        conn = get_connection()
        cursor = conn.cursor()

        # Insert health data
        cursor.execute("""
            INSERT INTO health_records
            (age, gender, fever, bp, sugar, oxygen)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (age, gender, fever, bp, sugar, oxygen))

        record_id = cursor.lastrowid
        print("Inserted health_records ID:", record_id)

        if record_id is None:
            raise Exception("record_id is None")

        # Insert prediction result
        cursor.execute("""
            INSERT INTO results
            (record_id, predicted_risk, confidence)
            VALUES (%s,%s,%s)
        """, (record_id, prediction, confidence))

        conn.commit()
        print("DB Commit Successful")

        # Fetch joined record
        cursor.execute("""
            SELECT h.age, h.gender, h.fever, h.bp, h.sugar, h.oxygen,
                   r.predicted_risk, r.confidence
            FROM health_records h
            JOIN results r ON h.id = r.record_id
            WHERE r.record_id = %s
            LIMIT 1
        """, (record_id,))

        row = cursor.fetchone()

    except Exception as e:
        if "conn" in locals():
            conn.rollback()
        print("DB ERROR:", type(e).__name__, e)
        return "Database error occurred", 500

    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()

    print("---- DB TRANSACTION END ----")

    # ---- 4. OPTIONAL CSV SAVE (LOCAL ONLY) ----
    try:
        os.makedirs("dataset", exist_ok=True)
        file_exists = os.path.isfile(CSV_PATH)

        with open(CSV_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "age", "gender", "fever",
                    "bp", "sugar", "oxygen", "risk"
                ])

            writer.writerow([
                age, gender, fever,
                bp, sugar, oxygen, prediction
            ])

        print("CSV saved")

    except Exception as e:
        print("CSV ERROR:", e)

    if row is None:
        return "No data found", 500

    db_data = {
        "age": row[0],
        "gender": "Male" if row[1] == 1 else "Female",
        "fever": "Yes" if row[2] == 1 else "No",
        "bp": row[3],
        "sugar": row[4],
        "oxygen": row[5],
        "result": risk_map[row[6]],
        "risk_class": risk_class_map[row[6]],
        "confidence": row[7]
    }

    return render_template("result.html", db_data=db_data)

# ---------- Run ----------
if __name__ == "__main__":
    app.run()
