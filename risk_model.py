"""
HORUS ICU Early Warning System — ML Risk Engine
================================================
Hackathon Rosetta Code · Round 2

Pipeline:
  1. Decode hex telemetry stream  →  raw vitals
  2. Resolve colliding ghost IDs  →  patient identity
  3. Decode scrambled drug names  →  Caesar cipher reversal
  4. Compute NEWS2 clinical score →  weighted vital scoring
  5. Isolation Forest             →  anomaly detection layer
  6. Fuse scores                  →  final 0–100 risk score
  7. Emit JSON                    →  dashboard-ready payload
"""

import json, math, random, string, time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

# ─── 1. LOAD REAL DATA ────────────────────────────────────────────────────────

demographics = pd.read_csv("patient_demographics.csv")
prescriptions = pd.read_csv("prescription_audit.csv")

# ─── 2. HEX TELEMETRY DECODER ─────────────────────────────────────────────────

def hex_to_vitals(hex_str: str, patient_age: int) -> dict:
    """
    Decode a 4-byte hex telemetry packet into vital signs.
    Encoding scheme (designed for this hackathon):
      byte0: heart rate offset   (base 60 + age_factor)
      byte1: systolic BP offset  (base 100)
      byte2: SpO2 offset         (base 85, inverted — lower byte = higher SpO2)
      byte3: respiratory rate    (base 12)
    """
    val = int(hex_str, 16)
    b0 = (val >> 24) & 0xFF
    b1 = (val >> 16) & 0xFF
    b2 = (val >> 8)  & 0xFF
    b3 =  val        & 0xFF

    age_factor = patient_age / 100.0
    hr   = int(60  + (b0 / 255.0) * 60  + age_factor * 20)
    bp   = int(100 + (b1 / 255.0) * 80  + age_factor * 30)
    spo2 = int(100 - (b2 / 255.0) * 20  - age_factor * 5)
    rr   = int(12  + (b3 / 255.0) * 20  + age_factor * 5)
    temp = round(36.0 + (b0 / 255.0) * 3.5 - (b2 / 255.0) * 1.0, 1)

    return {
        "hr":   max(40, min(200, hr)),
        "bp":   max(80, min(220, bp)),
        "spo2": max(80, min(100, spo2)),
        "rr":   max(8,  min(40,  rr)),
        "temp": max(35.0, min(41.0, temp)),
    }

def generate_hex_stream(patient_age: int, tick: float) -> str:
    """Simulate a live hex telemetry packet for a patient."""
    noise = math.sin(tick * 0.7) * 0.3 + math.sin(tick * 1.3) * 0.15
    age_norm = patient_age / 100.0

    b0 = int(((age_norm * 0.4 + abs(noise) * 0.6)) * 255) & 0xFF
    b1 = int(((age_norm * 0.5 + abs(noise) * 0.4)) * 255) & 0xFF
    b2 = int(((1.0 - age_norm * 0.3 - abs(noise) * 0.3)) * 255) & 0xFF
    b3 = int(((age_norm * 0.4 + abs(noise) * 0.3)) * 255) & 0xFF

    raw = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    return f"0x{raw:08X}"

# ─── 3. IDENTITY COLLISION RESOLVER ───────────────────────────────────────────

def resolve_identity(ghost_id: str, vitals: dict, demographics_df: pd.DataFrame) -> dict:
    """
    Two patients can share the same ghost_id (parity_group 0 and 1).
    We use vital sign parity to determine which patient is which:
      - parity_group 0: younger patient in the pair → lower HR variance expected
      - parity_group 1: older patient in the pair → higher age-weighted risk
    Returns the resolved patient record.
    """
    candidates = demographics_df[demographics_df["ghost_id"] == ghost_id]

    if len(candidates) == 1:
        row = candidates.iloc[0]
        return {"resolved": True, "collision": False,
                "patient": row["name"], "age": row["age"],
                "internal_id": int(row["internal_id"]), "parity": int(row["parity_group"])}

    if len(candidates) == 2:
        p0 = candidates[candidates["parity_group"] == 0].iloc[0]
        p1 = candidates[candidates["parity_group"] == 1].iloc[0]

        # Parity rule: higher HR + higher age → parity_group 1
        hr_parity = 1 if vitals["hr"] > 90 else 0
        age_parity = 1 if p1["age"] > p0["age"] else 0
        resolved_parity = 1 if (hr_parity + age_parity) >= 1 else 0

        chosen = p1 if resolved_parity == 1 else p0
        return {"resolved": True, "collision": True,
                "patient": chosen["name"], "age": int(chosen["age"]),
                "internal_id": int(chosen["internal_id"]),
                "parity": int(chosen["parity_group"]),
                "collision_partner": (p0 if resolved_parity == 1 else p1)["name"]}

    return {"resolved": False, "collision": False,
            "patient": ghost_id, "age": 50, "internal_id": -1, "parity": 0}

# ─── 4. CAESAR CIPHER DRUG DECODER ────────────────────────────────────────────

KNOWN_DRUGS = {
    "Amoxicillin", "Metformin", "Lisinopril", "Atorvastatin",
    "Omeprazole", "Amlodipine", "Metoprolol", "Furosemide",
    "Warfarin", "Insulin", "Morphine", "Heparin",
    "Clopidogrel", "Ramipril", "Digoxin", "Vancomycin",
}

HIGH_RISK_DRUGS = {"Warfarin", "Morphine", "Heparin", "Digoxin", "Vancomycin", "Insulin"}

def decode_drug(scrambled: str) -> dict:
    """
    Attempt Caesar cipher reversal across all 25 shifts.
    If no known drug is found, flag as unresolved (potential contraband/error).
    """
    # First check if it's already a real drug (like Amoxicillin in the data)
    if scrambled.capitalize() in KNOWN_DRUGS or scrambled in KNOWN_DRUGS:
        name = scrambled.capitalize()
        return {"decoded": name, "shift": 0, "resolved": True,
                "high_risk": name in HIGH_RISK_DRUGS}

    for shift in range(1, 26):
        candidate = ""
        for ch in scrambled:
            if ch.isalpha():
                base = ord('A') if ch.isupper() else ord('a')
                candidate += chr((ord(ch) - base - shift) % 26 + base)
            else:
                candidate += ch
        if candidate.capitalize() in KNOWN_DRUGS:
            name = candidate.capitalize()
            return {"decoded": name, "shift": shift, "resolved": True,
                    "high_risk": name in HIGH_RISK_DRUGS}

    # Phonetic heuristic: if unresolved, mark as suspicious
    return {"decoded": f"UNKNOWN({scrambled})", "shift": -1,
            "resolved": False, "high_risk": True}

# ─── 5. NEWS2 CLINICAL SCORING ────────────────────────────────────────────────

def news2_score(vitals: dict) -> dict:
    """
    National Early Warning Score 2 (NEWS2) — UK NHS standard.
    Returns component scores and total (0–20 range).
    Higher score = more critical.
    """
    scores = {}

    # Respiratory rate (breaths/min)
    rr = vitals["rr"]
    if rr <= 8:             scores["rr"] = 3
    elif rr <= 11:          scores["rr"] = 1
    elif rr <= 20:          scores["rr"] = 0
    elif rr <= 24:          scores["rr"] = 2
    else:                   scores["rr"] = 3

    # SpO2 (oxygen saturation %)
    spo2 = vitals["spo2"]
    if spo2 >= 96:          scores["spo2"] = 0
    elif spo2 >= 94:        scores["spo2"] = 1
    elif spo2 >= 92:        scores["spo2"] = 2
    else:                   scores["spo2"] = 3

    # Systolic BP (mmHg)
    bp = vitals["bp"]
    if bp <= 90:            scores["bp"] = 3
    elif bp <= 100:         scores["bp"] = 2
    elif bp <= 110:         scores["bp"] = 1
    elif bp <= 219:         scores["bp"] = 0
    else:                   scores["bp"] = 3

    # Heart rate (bpm)
    hr = vitals["hr"]
    if hr <= 40:            scores["hr"] = 3
    elif hr <= 50:          scores["hr"] = 1
    elif hr <= 90:          scores["hr"] = 0
    elif hr <= 110:         scores["hr"] = 1
    elif hr <= 130:         scores["hr"] = 2
    else:                   scores["hr"] = 3

    # Temperature (°C)
    temp = vitals["temp"]
    if temp <= 35.0:        scores["temp"] = 3
    elif temp <= 36.0:      scores["temp"] = 1
    elif temp <= 38.0:      scores["temp"] = 0
    elif temp <= 39.0:      scores["temp"] = 1
    else:                   scores["temp"] = 2

    total = sum(scores.values())

    # NEWS2 clinical interpretation
    if total >= 7:       level = "CRITICAL"
    elif total >= 5:     level = "HIGH"
    elif total >= 3:     level = "MEDIUM"
    else:                level = "LOW"

    return {"components": scores, "total": total, "level": level}

# ─── 6. ISOLATION FOREST ANOMALY LAYER ───────────────────────────────────────

def build_anomaly_model(n_patients: int = 200) -> tuple:
    """
    Train Isolation Forest on synthetic 'normal' ICU population.
    Returns (model, scaler) — used to score each incoming patient.
    """
    normal_data = []
    for _ in range(n_patients):
        normal_data.append([
            random.gauss(75, 12),   # HR — normal ICU range
            random.gauss(125, 18),  # BP
            random.gauss(96, 2),    # SpO2
            random.gauss(16, 3),    # RR
            random.gauss(37.0, 0.5),# Temp
        ])

    X = np.array(normal_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.08,   # ~8% of ICU patients expected anomalous
        random_state=42,
        max_samples="auto",
    )
    model.fit(X_scaled)
    return model, scaler

# ─── 7. FUSED RISK SCORE ─────────────────────────────────────────────────────

def compute_risk_score(
    vitals: dict,
    news2: dict,
    anomaly_score: float,
    patient_age: int,
    drug_risk: bool = False,
) -> dict:
    """
    Fuse NEWS2 + Isolation Forest + age + drug risk into a 0–100 score.

    Weights (tuned for clinical relevance):
      - NEWS2 contribution:      45%
      - Anomaly score:           30%
      - Age factor:              15%
      - Drug interaction risk:   10%
    """
    # NEWS2: max total is 20, normalize to 0–1
    news2_norm = min(news2["total"] / 20.0, 1.0)

    # Isolation Forest: returns negative value for anomalies; remap to 0–1
    # score_samples() returns log likelihood; more negative = more anomalous
    anomaly_norm = max(0.0, min(1.0, -anomaly_score))

    # Age factor: sigmoid centred at 70, steeper for elderly
    age_norm = 1 / (1 + math.exp(-(patient_age - 68) / 8.0))

    # Drug risk binary
    drug_norm = 1.0 if drug_risk else 0.0

    # Weighted fusion
    risk = (
        news2_norm  * 0.45 +
        anomaly_norm * 0.30 +
        age_norm    * 0.15 +
        drug_norm   * 0.10
    )

    score = int(min(99, max(1, round(risk * 100))))

    if score >= 75:   status = "CRITICAL"
    elif score >= 50: status = "HIGH"
    elif score >= 30: status = "MEDIUM"
    else:             status = "STABLE"

    return {
        "score": score,
        "status": status,
        "components": {
            "news2_weight":   round(news2_norm  * 0.45 * 100, 1),
            "anomaly_weight": round(anomaly_norm * 0.30 * 100, 1),
            "age_weight":     round(age_norm    * 0.15 * 100, 1),
            "drug_weight":    round(drug_norm   * 0.10 * 100, 1),
        }
    }

# ─── 8. FULL PIPELINE — GENERATE PATIENT SNAPSHOT ────────────────────────────

def run_pipeline(tick: float, model: IsolationForest, scaler: StandardScaler) -> dict:
    """
    Run the full HORUS pipeline for all patients at a given time tick.
    Returns a dashboard-ready JSON payload.
    """
    results = []

    # Get all unique ghost IDs from the stream
    ghost_ids = demographics["ghost_id"].unique()

    for ghost_id in ghost_ids[:20]:  # ICU capacity: 20 bays
        # Pick one patient from this ghost_id group for the stream
        group = demographics[demographics["ghost_id"] == ghost_id]
        seed_patient = group.iloc[0]

        # Step 1: Generate hex telemetry
        hex_packet = generate_hex_stream(seed_patient["age"], tick + seed_patient["internal_id"] * 0.7)

        # Step 2: Decode vitals from hex
        vitals = hex_to_vitals(hex_packet, seed_patient["age"])

        # Step 3: Resolve identity collision
        identity = resolve_identity(ghost_id, vitals, demographics)

        # Step 4: Decode prescriptions for this ghost_id
        rxs = prescriptions[prescriptions["ghost_id"] == ghost_id]
        decoded_drugs = [decode_drug(row["scrambled_med"]) for _, row in rxs.iterrows()]
        drug_risk = any(d["high_risk"] for d in decoded_drugs)

        # Step 5: NEWS2 score
        news = news2_score(vitals)

        # Step 6: Isolation Forest anomaly score
        features = np.array([[vitals["hr"], vitals["bp"], vitals["spo2"], vitals["rr"], vitals["temp"]]])
        features_scaled = scaler.transform(features)
        raw_anomaly = float(model.score_samples(features_scaled)[0])

        # Step 7: Fuse into final risk score
        risk = compute_risk_score(vitals, news, raw_anomaly, identity["age"], drug_risk)

        results.append({
            "ghost_id":      ghost_id,
            "hex_packet":    hex_packet,
            "identity":      identity,
            "vitals":        vitals,
            "news2":         news,
            "drugs":         decoded_drugs,
            "drug_risk":     drug_risk,
            "anomaly_raw":   round(raw_anomaly, 4),
            "risk":          risk,
            "tick":          round(tick, 2),
        })

    # Sort by risk score descending
    results.sort(key=lambda x: x["risk"]["score"], reverse=True)
    return results

# ─── 9. DEMO RUN ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building anomaly detection model...")
    model, scaler = build_anomaly_model(n_patients=300)
    print("Model trained on 300 synthetic ICU patients\n")

    print("Running pipeline at tick=0.0...\n")
    results = run_pipeline(tick=0.0, model=model, scaler=scaler)

    print(f"{'Patient':<14} {'Age':>4} {'Ghost ID':<10} {'HR':>4} {'BP':>4} {'SpO2':>5} "
          f"{'RR':>4} {'NEWS2':>6} {'Anomaly':>8} {'Risk':>5} {'Status':<10} {'Drugs'}")
    print("─" * 110)

    for p in results:
        v = p["vitals"]
        drugs = ", ".join(d["decoded"] for d in p["drugs"][:2])
        collision = " ⚡COLLISION" if p["identity"].get("collision") else ""
        print(
            f"{p['identity']['patient']:<14} {p['identity']['age']:>4} "
            f"{p['ghost_id']:<10} {v['hr']:>4} {v['bp']:>4} {v['spo2']:>5}% "
            f"{v['rr']:>4} {p['news2']['total']:>6} {p['anomaly_raw']:>8.3f} "
            f"{p['risk']['score']:>5} {p['risk']['status']:<10} {drugs}{collision}"
        )

    print(f"\n{'─'*110}")
    critical = [p for p in results if p["risk"]["status"] == "CRITICAL"]
    print(f"CRITICAL patients: {len(critical)}")
    collisions = [p for p in results if p["identity"].get("collision")]
    print(f"Identity collisions resolved: {len(collisions)}")
    high_rx = [p for p in results if p["drug_risk"]]
    print(f"High-risk drug flags: {len(high_rx)}")

    # Save JSON payload for dashboard
    with open("icu_payload.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nPayload saved to icu_payload.json")