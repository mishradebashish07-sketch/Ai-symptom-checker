import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="AI Symptom Checker", page_icon="🩺", layout="centered"
)

st.title("🩺 Medical AI Symptom Checker")
st.write(
    "Describe how you are feeling below, or select specific symptoms to see the top possible conditions."
)


# ==========================================
# 2. CREATE OR LOAD DATASET & TRAIN MODEL
# ==========================================
@st.cache_resource
def load_and_train_model():
    csv_filename = "Training.csv"

    # If Training.csv doesn't exist yet, generate a clean local medical dataset automatically!
    if not os.path.exists(csv_filename):
        symptoms = [
            "high_fever",
            "chills",
            "cough",
            "runny_nose",
            "continuous_sneezing",
            "breathlessness",
            "stomach_pain",
            "vomiting",
            "nausea",
            "headache",
            "fatigue",
            "itching",
            "skin_rash",
            "joint_pain",
            "diarrhoea",
            "chest_pain",
            "dizziness",
            "sweating",
            "loss_of_appetite",
            "muscle_pain",
        ]

        # Disease profiles with their characteristic symptoms
        disease_profiles = {
            "Common Cold": [
                "cough",
                "runny_nose",
                "continuous_sneezing",
                "headache",
                "fatigue",
            ],
            "Dengue": [
                "high_fever",
                "headache",
                "joint_pain",
                "muscle_pain",
                "skin_rash",
                "vomiting",
                "fatigue",
            ],
            "Malaria": [
                "high_fever",
                "chills",
                "sweating",
                "headache",
                "nausea",
                "vomiting",
                "muscle_pain",
            ],
            "Typhoid": [
                "high_fever",
                "stomach_pain",
                "headache",
                "fatigue",
                "diarrhoea",
                "loss_of_appetite",
            ],
            "Gastroenteritis (Food Poisoning)": [
                "stomach_pain",
                "vomiting",
                "nausea",
                "diarrhoea",
                "fatigue",
            ],
            "Fungal Infection / Allergy": ["itching", "skin_rash"],
            "Migraine": ["headache", "nausea", "dizziness", "vomiting"],
            "Bronchial Asthma / Pneumonia": [
                "cough",
                "breathlessness",
                "high_fever",
                "chest_pain",
                "fatigue",
            ],
            "Heart Attack (Emergency Warning)": [
                "chest_pain",
                "breathlessness",
                "sweating",
                "dizziness",
                "nausea",
            ],
            "Hypertension / Vertigo": ["headache", "dizziness", "chest_pain"],
        }

        rows = []
        np.random.seed(42)
        # Generate 500 realistic patient training records
        for disease, active_syms in disease_profiles.items():
            for _ in range(50):
                row = {sym: 0 for sym in symptoms}
                for s in active_syms:
                    row[s] = 1 if np.random.rand() > 0.15 else 0
                row["prognosis"] = disease
                rows.append(row)

        pd.DataFrame(rows).to_csv(csv_filename, index=False)

    df = pd.read_csv(csv_filename)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    target_column = "prognosis"
    symptoms_list = [col for col in df.columns if col != target_column]

    X = df[symptoms_list]
    y = df[target_column]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, symptoms_list, len(df), len(y.unique())


model, symptoms_list, total_records, total_diseases = load_and_train_model()

st.caption(
    f"✅ AI Model Ready: Trained on **{total_records}** patient records across **{total_diseases}** conditions and **{len(symptoms_list)}** symptoms."
)


# ==========================================
# 3. SYMPTOM EXTRACTOR FUNCTION
# ==========================================
def extract_symptoms_from_text(user_input):
    user_input = user_input.lower()
    extracted = []

    for symptom in symptoms_list:
        clean_symptom = symptom.replace("_", " ").strip()
        if clean_symptom in user_input:
            extracted.append(symptom)

    slang_map = {
        "high_fever": ["fever", "hot", "temperature", "burning up"],
        "chills": ["chills", "shivering", "freezing"],
        "cough": ["cough", "coughing", "hack"],
        "runny_nose": ["runny nose", "stuffy", "mucus", "cold"],
        "continuous_sneezing": ["sneezing", "sneeze"],
        "breathlessness": [
            "shortness of breath",
            "cant breathe",
            "can't breathe",
            "breathless",
            "breathing",
        ],
        "stomach_pain": [
            "stomach",
            "belly",
            "tummy",
            "cramps",
            "gut",
            "abdominal",
        ],
        "vomiting": ["vomit", "vomiting", "throw up", "throwing up", "puking"],
        "nausea": ["nausea", "nauseous", "feel sick"],
        "headache": ["headache", "head hurts", "migraine"],
        "fatigue": ["tired", "exhausted", "weak", "no energy", "fatigue"],
        "itching": ["itch", "itchy", "scratching"],
        "skin_rash": ["rash", "red spots", "bumps"],
        "joint_pain": ["joint", "knees hurt", "elbows hurt"],
        "muscle_pain": ["body ache", "muscle", "sore body"],
        "diarrhoea": ["diarrhea", "loose motion", "watery stool"],
        "chest_pain": ["chest pain", "chest hurts", "heavy chest"],
        "dizziness": ["dizzy", "spinning", "lightheaded"],
        "sweating": ["sweat", "sweating"],
        "loss_of_appetite": ["not hungry", "no appetite", "cant eat"],
    }

    for csv_symptom, phrases in slang_map.items():
        if csv_symptom in symptoms_list and csv_symptom not in extracted:
            for phrase in phrases:
                if phrase in user_input:
                    extracted.append(csv_symptom)
                    break

    return extracted


# ==========================================
# 4. WEBSITE USER INTERFACE
# ==========================================
st.subheader("1. Describe Your Symptoms")
user_text = st.text_area(
    "Type in plain English (e.g., 'I have a high fever, terrible headache, joint pain, and chills'):",
    height=100,
)

st.subheader("2. Or Select Additional Symptoms Manually (Optional)")
clean_to_raw = {s.replace("_", " ").strip().title(): s for s in symptoms_list}
selected_clean = st.multiselect(
    "Search and pick symptoms from the database:",
    options=list(clean_to_raw.keys()),
)

if st.button("🔍 Analyze Symptoms", type="primary"):
    found_from_text = (
        extract_symptoms_from_text(user_text) if user_text else []
    )
    found_from_dropdown = [clean_to_raw[c] for c in selected_clean]

    final_symptoms = list(set(found_from_text + found_from_dropdown))

    if not final_symptoms:
        st.error(
            "❌ No symptoms detected! Please describe your symptoms or pick some from the dropdown list."
        )
    else:
        readable_symptoms = [
            s.replace("_", " ").strip().title() for s in final_symptoms
        ]
        st.success(f"**🧠 Detected Symptoms:** {', '.join(readable_symptoms)}")

        input_vector = np.zeros(len(symptoms_list))
        for symptom in final_symptoms:
            if symptom in symptoms_list:
                idx = symptoms_list.index(symptom)
                input_vector[idx] = 1

        input_df = pd.DataFrame([input_vector], columns=symptoms_list)

        # Grab [0] so we get the list of probabilities for this single patient
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        results = sorted(
            zip(classes, probabilities), key=lambda x: x[1], reverse=True
        )

        st.markdown("### 🏥 Top Possible Conditions")
        for disease, prob in results[:5]:
            if prob > 0:
                percentage = prob * 100
                st.write(f"**{disease}** — `{percentage:.1f}% Match`")
                st.progress(float(prob))

        st.warning(
            "⚠️ **Medical Disclaimer:** This is a student AI trial prototype for educational purposes only. Always consult a qualified doctor for medical advice."
        )
