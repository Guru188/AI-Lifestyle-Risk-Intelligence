from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import webbrowser
import threading

app = Flask(__name__)

# ── Load models (each pkl is {"model": ..., "features": [...]}) ──
with open("models/diabetes_model.pkl", "rb") as f:
    d_pkg = pickle.load(f)
    diabetes_model    = d_pkg["model"]
    diabetes_features = d_pkg["features"]

with open("models/stress_model.pkl", "rb") as f:
    s_pkg = pickle.load(f)
    stress_model    = s_pkg["model"]
    stress_features = s_pkg["features"]

with open("models/hypertension_model.pkl", "rb") as f:
    h_pkg = pickle.load(f)
    hyper_model    = h_pkg["model"]
    hyper_features = h_pkg["features"]


def compute_risks(age, bmi, sleep, water, screen_time, exercise, diet, glucose):
    diet_map = {"healthy": 0, "mixed": 1, "junk": 2}

    # Estimate heart rate & steps from app inputs
    heart_rate = 80 - (exercise * 0.1)           # more exercise → lower resting HR
    daily_steps = exercise * 100                  # rough estimate
    heart_rate = max(55, min(100, heart_rate))
    daily_steps = max(1000, min(15000, daily_steps))

    # ── Diabetes features: Glucose, BMI, Age, Insulin, BloodPressure ──
    insulin = max(0, 200 - exercise * 1.2)        # more exercise → lower insulin resistance
    systolic_bp = 110 + (bmi - 22) * 0.8 + (age - 30) * 0.3
    d_input = np.array([[glucose, bmi, age, insulin, systolic_bp]])
    diabetes_prob = diabetes_model.predict_proba(d_input)[0][1]

    # ── Stress/Hyper features: Sleep Duration, Physical Activity Level,
    #                           Heart Rate, Daily Steps, BMI_num, Age ──
    sh_input = np.array([[sleep, exercise, heart_rate, daily_steps, bmi, age]])
    stress_prob = stress_model.predict_proba(sh_input)[0][1]
    hyper_prob  = hyper_model.predict_proba(sh_input)[0][1]

    # Weighted average risk (0-1), then invert to get health score
    avg_risk = (diabetes_prob * 0.40) + (stress_prob * 0.35) + (hyper_prob * 0.25)
    health_score = round(100 - (avg_risk * 100))
    health_score = max(5, min(100, health_score))  # minimum 5 so ring is always visible

    return {
        "diabetes":     round(diabetes_prob * 100, 1),
        "stress":       round(stress_prob * 100, 1),
        "hypertension": round(hyper_prob * 100, 1),
        "health_score": health_score
    }


def get_explanations(age, bmi, sleep, water, screen_time, exercise, diet, glucose):
    reasons = []
    if sleep < 6:
        reasons.append(f"Low sleep ({sleep} hrs/night) significantly raises stress & hypertension risk")
    if screen_time > 8:
        reasons.append(f"High screen time ({screen_time} hrs) is linked to elevated stress levels")
    if exercise < 30:
        reasons.append(f"Low physical activity ({exercise} mins/day) raises diabetes & hypertension risk")
    if bmi > 30:
        reasons.append(f"High BMI ({bmi}) is a leading driver of diabetes and hypertension")
    if water < 1.5:
        reasons.append(f"Low water intake ({water}L/day) affects metabolism and kidney function")
    if glucose > 140:
        reasons.append(f"Elevated fasting glucose ({glucose} mg/dL) is a direct diabetes risk factor")
    if diet == "junk":
        reasons.append("Junk diet increases insulin resistance and inflammation across all 3 risks")
    if age > 45:
        reasons.append(f"Age ({age}) is a natural risk factor — regular screening is recommended")
    if not reasons:
        reasons.append("Your lifestyle inputs are in a healthy range — keep maintaining these habits!")
    return reasons


def get_action_plan(sleep, water, screen_time, exercise, diet):
    actions = []
    if sleep < 7:
        actions.append({"icon": "🌙", "tip": f"Increase sleep to 7–8 hrs (you currently get {sleep} hrs)"})
    if screen_time > 6:
        actions.append({"icon": "📵", "tip": f"Reduce screen time by 1–2 hrs/day (currently {screen_time} hrs)"})
    if exercise < 30:
        actions.append({"icon": "🚶", "tip": "Walk at least 6,000 steps/day for the next 7 days"})
    if water < 2:
        actions.append({"icon": "💧", "tip": f"Increase water to 2.5L/day (currently {water}L)"})
    if diet == "junk":
        actions.append({"icon": "🥗", "tip": "Replace one junk meal daily with fruits or salad"})
    if not actions:
        actions.append({"icon": "✅", "tip": "Great lifestyle! Maintain your current habits."})
    return actions


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data        = request.form
    age         = float(data.get("age", 25))
    bmi         = float(data.get("bmi", 22))
    sleep       = float(data.get("sleep", 7))
    water       = float(data.get("water", 2))
    screen_time = float(data.get("screen_time", 4))
    exercise    = float(data.get("exercise", 30))
    diet        = data.get("diet", "mixed")
    glucose     = float(data.get("glucose", 100))

    risks   = compute_risks(age, bmi, sleep, water, screen_time, exercise, diet, glucose)
    reasons = get_explanations(age, bmi, sleep, water, screen_time, exercise, diet, glucose)
    actions = get_action_plan(sleep, water, screen_time, exercise, diet)

    return render_template("result.html",
        risks=risks, reasons=reasons, actions=actions,
        inputs={"age": age, "bmi": bmi, "sleep": sleep, "water": water,
                "screen_time": screen_time, "exercise": exercise,
                "diet": diet, "glucose": glucose}
    )


@app.route("/simulate", methods=["POST"])
def simulate():
    data        = request.json
    age         = float(data.get("age", 25))
    bmi         = float(data.get("bmi", 22))
    sleep       = float(data.get("sleep", 7))
    water       = float(data.get("water", 2))
    screen_time = float(data.get("screen_time", 4))
    exercise    = float(data.get("exercise", 30))
    diet        = data.get("diet", "mixed")
    glucose     = float(data.get("glucose", 100))

    risks = compute_risks(age, bmi, sleep, water, screen_time, exercise, diet, glucose)
    return jsonify(risks)


if __name__ == "__main__":
    # Auto-open browser after 1 second (gives Flask time to start)
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    print("\n🚀 Starting server... Browser will open automatically!")
    app.run(debug=False)
