import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import urllib.request

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# AUTO-DOWNLOAD DATASETS
# ─────────────────────────────────────────────────────────────

PIMA_URL  = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
SLEEP_URL = "https://gist.github.com/sanjanaadapa7/db0cbec1ec87724ef6be3198b6f9f0cc/raw/5f1f8ca90f428420e2df3e551a15068b0fc04cff/Sleep_health_and_lifestyle_dataset.csv"

PIMA_PATH  = "data/diabetes.csv"
SLEEP_PATH = "data/sleep_health.csv"

print("📥 Downloading PIMA Diabetes dataset...")
try:
    urllib.request.urlretrieve(PIMA_URL, PIMA_PATH)
    print("   ✅ PIMA dataset downloaded!")
except Exception as e:
    print(f"   ⚠️  Download failed ({e}), using synthetic fallback...")
    PIMA_PATH = None

print("📥 Downloading Sleep Health & Lifestyle dataset...")
try:
    urllib.request.urlretrieve(SLEEP_URL, SLEEP_PATH)
    print("   ✅ Sleep Health dataset downloaded!")
except Exception as e:
    print(f"   ⚠️  Download failed ({e}), using synthetic fallback...")
    SLEEP_PATH = None


# ─────────────────────────────────────────────────────────────
# 1. DIABETES MODEL — PIMA Dataset
# ─────────────────────────────────────────────────────────────
print("\n🍬 Training Diabetes model...")

if PIMA_PATH and os.path.exists(PIMA_PATH):
    df = pd.read_csv(PIMA_PATH)
    print(f"   📊 Using real PIMA dataset — {len(df)} rows")
    df = df.rename(columns={"Outcome": "label"})
    df = df[df["Glucose"] > 0]
    df = df[df["BMI"] > 0]
    features = ["Glucose", "BMI", "Age", "Insulin", "BloodPressure"]
    X = df[features]
    y = df["label"]
else:
    print("   📊 Using synthetic fallback data")
    np.random.seed(42)
    n = 1000
    glucose = np.random.uniform(70, 200, n)
    bmi     = np.random.uniform(18, 45, n)
    age     = np.random.randint(18, 70, n)
    insulin = np.random.uniform(0, 300, n)
    bp      = np.random.uniform(60, 120, n)
    risk    = ((glucose > 140)*3 + (bmi > 30)*2 + (age > 45)*1).astype(int)
    X = pd.DataFrame({"Glucose": glucose, "BMI": bmi, "Age": age,
                      "Insulin": insulin, "BloodPressure": bp})
    y = (risk >= 4).astype(int)
    features = ["Glucose", "BMI", "Age", "Insulin", "BloodPressure"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
dm = RandomForestClassifier(n_estimators=150, random_state=42)
dm.fit(X_tr, y_tr)
print(f"   ✅ Accuracy: {dm.score(X_te, y_te):.2%}")
with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump({"model": dm, "features": features}, f)


# ─────────────────────────────────────────────────────────────
# 2. STRESS + HYPERTENSION — Sleep Health Dataset
# ─────────────────────────────────────────────────────────────
print("\n🧠 Training Stress & Hypertension models...")

if SLEEP_PATH and os.path.exists(SLEEP_PATH):
    df2 = pd.read_csv(SLEEP_PATH)
    print(f"   📊 Using real Sleep Health dataset — {len(df2)} rows")

    # Map BMI Category text → numeric
    df2["BMI_num"] = df2["BMI Category"].map(
        {"Normal": 22, "Normal Weight": 22, "Overweight": 28, "Obese": 35}
    ).fillna(24)

    # Stress label: Stress Level > 6 = high stress
    df2["stress_label"] = (df2["Stress Level"] > 6).astype(int)

    # Hypertension label: systolic BP > 130
    def parse_bp(bp_str):
        try: return int(str(bp_str).split("/")[0])
        except: return 120
    df2["systolic"]    = df2["Blood Pressure"].apply(parse_bp)
    df2["hyper_label"] = (df2["systolic"] > 130).astype(int)

    sleep_features = ["Sleep Duration", "Physical Activity Level",
                      "Heart Rate", "Daily Steps", "BMI_num", "Age"]
    df2 = df2.dropna(subset=sleep_features)
    X2  = df2[sleep_features]

else:
    print("   📊 Using synthetic fallback data")
    np.random.seed(99)
    n = 1000
    sleep      = np.random.uniform(3, 10, n)
    activity   = np.random.uniform(0, 90, n)
    heart_rate = np.random.uniform(55, 100, n)
    steps      = np.random.uniform(2000, 15000, n)
    bmi_num    = np.random.uniform(18, 40, n)
    age        = np.random.randint(18, 65, n)
    X2 = pd.DataFrame({"Sleep Duration": sleep, "Physical Activity Level": activity,
                        "Heart Rate": heart_rate, "Daily Steps": steps,
                        "BMI_num": bmi_num, "Age": age})
    stress_risk = ((sleep < 6)*3 + (activity < 20)*2 + (heart_rate > 85)*1).astype(int)
    hyper_risk  = ((bmi_num > 28)*3 + (sleep < 6)*2 + (activity < 20)*2).astype(int)
    df2 = pd.DataFrame(X2)
    df2["stress_label"] = (stress_risk >= 4).astype(int)
    df2["hyper_label"]  = (hyper_risk >= 4).astype(int)
    sleep_features = ["Sleep Duration", "Physical Activity Level",
                       "Heart Rate", "Daily Steps", "BMI_num", "Age"]

# ── Stress Model ──
y_stress = df2["stress_label"]
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X2, y_stress, test_size=0.2, random_state=42)
sm = RandomForestClassifier(n_estimators=150, random_state=42)
sm.fit(Xs_tr, ys_tr)
print(f"   ✅ Stress Accuracy: {sm.score(Xs_te, ys_te):.2%}")
with open("models/stress_model.pkl", "wb") as f:
    pickle.dump({"model": sm, "features": sleep_features}, f)

# ── Hypertension Model ──
y_hyper = df2["hyper_label"]
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(X2, y_hyper, test_size=0.2, random_state=42)
hm = RandomForestClassifier(n_estimators=150, random_state=42)
hm.fit(Xh_tr, yh_tr)
print(f"   ✅ Hypertension Accuracy: {hm.score(Xh_te, yh_te):.2%}")
with open("models/hypertension_model.pkl", "wb") as f:
    pickle.dump({"model": hm, "features": sleep_features}, f)

print("\n✅ All 3 models trained and saved in /models folder!")
print("📁 Real datasets saved in /data folder!")
print("\n🚀 Now run: python app.py")
