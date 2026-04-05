# 🧠 HORUS — ICU Early Warning System

> 🚨 Hackathon Rosetta Code · Round 2  
> ⚡ Real-time AI-powered system for predicting critical medical emergencies from streaming ICU data  

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Core Pipeline](#-core-pipeline)
- [Key Components](#-key-components)
- [Machine Learning Model](#-machine-learning-model)
- [Risk Scoring Logic](#-risk-scoring-logic)
- [Dashboard Features](#-dashboard-features)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Sample Output](#-sample-output)
- [Tech Stack](#-tech-stack)
- [Challenges Solved](#-challenges-solved)
- [Future Improvements](#-future-improvements)
- [Conclusion](#-conclusion)

---

## 🚨 Overview

**HORUS** is a real-time ICU monitoring and early warning system that predicts critical patient deterioration **before it happens** using streaming telemetry data, clinical scoring systems, and machine learning.

The system processes **high-velocity, obfuscated hexadecimal data streams**, resolves patient identity collisions, and computes a **real-time risk score (0–100)** for each patient.

---

## ❗ Problem Statement

Modern ICUs generate massive amounts of patient data, but:

- Data streams are **noisy and encoded**
- Patient identities can **collide or overlap**
- Critical conditions often go unnoticed until it's too late

👉 The challenge:
> Build a system that can **decode, identify, analyze, and predict patient risk in real-time**

---

## 🏗️ System Architecture
    ┌──────────────┐
    │ Hex Stream   │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Decoder      │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Identity     │
    │ Resolver     │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Drug Decoder │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ NEWS2 Score  │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ ML Model     │
    │ (Isolation   │
    │  Forest)     │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Risk Fusion  │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Dashboard    │
    └──────────────┘
    
---

## 🔑 Key Components

### 📡 1. Hex Telemetry Decoder
- Converts 4-byte hex packets into:
  - Heart Rate (HR)
  - Blood Pressure (BP)
  - SpO₂
  - Respiratory Rate (RR)
  - Temperature

---

### 🧬 2. Identity Resolver
- Handles **ghost ID collisions**
- Uses:
  - Vital sign patterns
  - Age-based parity logic

👉 Ensures correct patient tracking in real-time

---

### 💊 3. Drug Decoder (Security Layer)
- Decodes Caesar cipher–encoded prescriptions
- Identifies:
  - Valid medications
  - Suspicious/unknown drugs
  - High-risk drugs

---

### 🏥 4. Clinical Scoring (NEWS2)

Implements NHS **NEWS2 standard**:

| Parameter | Scored |
|----------|--------|
| Respiratory Rate | ✅ |
| SpO₂ | ✅ |
| Blood Pressure | ✅ |
| Heart Rate | ✅ |
| Temperature | ✅ |

👉 Output:
- Score (0–20)
- Severity Level

---

## 🤖 Machine Learning Model

### Isolation Forest

- Unsupervised anomaly detection
- Trained on synthetic ICU data
- Detects deviations from normal patient behavior

**Why Isolation Forest?**
- Works without labeled data
- Fast for real-time systems
- Handles multidimensional features

---

## ⚡ Risk Scoring Logic

Final risk score is computed as:

### Output:

| Score Range | Status |
|------------|--------|
| 0–29 | STABLE |
| 30–49 | MEDIUM |
| 50–74 | HIGH |
| 75–100 | CRITICAL |

---

## 📊 Dashboard Features

### 🔥 Real-Time Monitoring
- Live risk trends
- Vital tracking

### 🛏 ICU Bed Map
- Visual patient allocation
- Color-coded severity

### 🚨 Alert Feed
- Instant alerts for:
  - Critical vitals
  - High-risk drugs
  - Anomalies

### 📈 Charts & Visualizations
- Risk time-series graph
- Vitals radar chart
- Anomaly scatter plot

### 🧾 Timeline
- Patient event history

---

## 🏗️ Project Structure

---

## ⚡ Installation & Setup

### 1. Clone repo

### 2. Install dependencies

### 3. Run pipeline

### 4. Launch dashboard
Open:

---

## ▶️ Usage

1. Run the backend script  
2. Generate ICU data (`icu_payload.json`)  
3. Open dashboard  
4. Monitor:
   - Patient risk
   - Alerts
   - Trends  

---

## 📦 Sample Output

---

## 🧪 Tech Stack

### Backend
- Python
- NumPy
- Pandas
- Scikit-learn

### Frontend
- HTML/CSS/JS
- Chart.js

---

## 🧩 Challenges Solved

✔ Real-time streaming data handling  
✔ Identity collision resolution  
✔ Clinical + ML hybrid model  
✔ Multi-source data fusion  
✔ Interactive ICU visualization  

---

## 🚀 Future Improvements

- LSTM / Transformer-based time-series models  
- Integration with real ICU devices  
- Cloud deployment (AWS/GCP)  
- Multi-hospital scaling  

---

## 🏁 Conclusion

HORUS demonstrates how **AI + real-time systems + clinical intelligence** can be combined to build a **scalable ICU early warning system** capable of saving lives through proactive intervention.

