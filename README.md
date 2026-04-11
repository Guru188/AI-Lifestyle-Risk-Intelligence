# 🧠 AI Lifestyle Risk Intelligence System

> *Know your disease risk before symptoms appear — understand why, and simulate how small changes save your future.*

Built for **APOGEE Innovation Challenge '26** | Sponsored by **BITS Pilani**

---

## 🚀 What It Does

Most health apps just track data. This app gives you **intelligent, explainable risk insights** across 3 diseases simultaneously — powered by real ML models trained on real datasets.

| Feature | Description |
|---|---|
| 📊 Multi-Risk Prediction | Diabetes, Stress, Hypertension — all at once |
| 🧠 Explainable AI | Tells you *why* you're at risk, not just the number |
| 🎮 Health Score | 0–100 score with color-coded status |
| 🔮 Future Simulation | Move sliders → see how your score changes live |
| 🎯 Action Plan | Specific, personalized steps (not generic advice) |
| 📈 Before vs After Chart | Visual proof of lifestyle impact |

---

## 🗄️ Datasets Used

| Dataset | Source | Used For |
|---|---|---|
| PIMA Indians Diabetes Dataset | UCI / Plotly GitHub | Diabetes risk model |
| Sleep Health & Lifestyle Dataset | Kaggle / GitHub Gist | Stress + Hypertension models |

---

## 🤖 ML Architecture

- **Model:** Random Forest Classifier (scikit-learn)
- **3 separate models** — one per disease
- **Explainability:** Rule-based XAI explaining top risk factors
- **Simulation:** Real-time re-inference with modified inputs (no retraining)
- **Health Score Formula:**
```
Health Score = 100 − (Diabetes Risk × 0.40 + Stress Risk × 0.35 + Hypertension Risk × 0.25) × 100
```

---

## 📁 Project Structure

```
AI-Lifestyle-Risk-Intelligence/
│
├── app.py                  # Flask backend
├── train_model.py          # Train & save all 3 models
├── requirements.txt        # Dependencies
│
├── models/                 # Auto-generated after training
│   ├── diabetes_model.pkl
│   ├── stress_model.pkl
│   └── hypertension_model.pkl
│
├── data/                   # Auto-downloaded datasets
│   ├── diabetes.csv
│   └── sleep_health.csv
│
└── templates/
    ├── index.html          # Input form
    └── result.html         # Dashboard output
```

---

## ⚙️ How To Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/AI-Lifestyle-Risk-Intelligence.git
cd AI-Lifestyle-Risk-Intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models (downloads datasets automatically)
```bash
python train_model.py
```

### 4. Run the app
```bash
python app.py
```

Browser opens automatically at `http://127.0.0.1:5000` 🚀

---

## 🌐 App Flow

```
User enters lifestyle data (age, BMI, sleep, exercise, diet...)
                    ↓
        Flask receives POST request
                    ↓
    Random Forest → 3 risk probabilities
                    ↓
    Health Score calculated (weighted formula)
                    ↓
Result page: Gauges + Score Ring + Chart + Explanations + Simulation
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn (Random Forest) |
| Frontend | HTML, CSS, JavaScript |
| Charts | Chart.js |
| Data | pandas, numpy |

