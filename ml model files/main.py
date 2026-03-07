
from fastapi import FastAPI
from pydantic import create_model, Field
import pandas as pd
import numpy as np
import joblib
import json

app = FastAPI(
    title="Insomnia Prediction API - By James Lequang",
    summary="Recall-Optimized Machine Learning Model for Insomnia Diagnosis",
    description="""
Hello, my name is James Lequang, and I am the creator of this diagnostic tool for insomnia. Below you can click on the Post/Predict button to use the tool to 
see if the model believes you have insomnia. In addition, you can view all of the input that the model will ask you in the Schema of the Post/Predict section and 
in the Features section. The tool is already pre-set to the average value from the NHANES 2017-March 2020 Prepandemic Files used in this model. To receive the most 
accurate score from this model as possible feel free to change the input values to match your circumstance. However, I understand that taking the time to fill out much
of this data could be tedious, so I would recommend filling out the input for the following: Depression, PHQ9 Total, Age, Weight, Waist Circumference, Weekday and 
Weekend Sleep Hours, and Caffeine Intake. This would result in a quick indicator to see if you have insomnia. Of course please be wary that this model is not always
correct. The model has about a 95% accuracy at diagnosing insomnia, and my purpose was to give people the chance to quickly self-diagnose themselves to see whether they
should contact a medical professional. So please do not take the results too seriously and always follow the doctor's advice! Also the output will list the reasons 
behind its output at the Top SHAP Features section I hope that you find this tool helpful!
""",
)

model = joblib.load("xgb_insomnia_model.joblib")
explainer = joblib.load("shap_explainer.joblib")

with open("feature_names.json") as f:
    feature_names = json.load(f)

with open("feature_means.json") as f:
    feature_means = json.load(f)

with open("best_threshold_f2.json") as f:
    thresh_data = json.load(f)
    best_threshold = thresh_data["best_threshold_f2"]

print(f"Loaded model with {len(feature_names)} features")
print(f"F2 threshold = {best_threshold:.4f}")

feature_descriptions = {

    "RIDAGEYR": "Age in years",
    "RIAGENDR": "Gender (1=Male, 2=Female)",
    "INDFMPIR": "Family income-to-poverty ratio",
    "weekday_sleep_hours": "Average weekday sleep hours",
    "weekend_sleep_hours": "Average weekend sleep hours",
    "SLD012": "Trouble falling asleep (0=No, 1=Yes)",
    "SLD013": "Trouble staying asleep (0=No, 1=Yes)",
    "SMD460": "Ever told had trouble sleeping",
    "SMD470": "Ever told had insomnia",
    "PHQ9_total": "Total PHQ-9 depression score",
    "PHQ9_Q1": "PHQ-9: Little interest/pleasure",
    "PHQ9_Q2": "PHQ-9: Feeling down/depressed",
    "PHQ9_Q3": "PHQ-9: Trouble sleeping",
    "PHQ9_Q4": "PHQ-9: Feeling tired",
    "PHQ9_Q5": "PHQ-9: Poor appetite/overeating",
    "PHQ9_Q6": "PHQ-9: Feeling bad about self",
    "PHQ9_Q7": "PHQ-9: Trouble concentrating",
    "PHQ9_Q8": "PHQ-9: Moving/speaking slowly",
    "PHQ9_Q9": "PHQ-9: Thoughts of self-harm",
    "BMXBMI": "Body Mass Index (BMI)",
    "BMXWAIST": "Waist circumference (cm)",
    "BPXSY1": "Systolic blood pressure (mmHg)",
    "BPXDI1": "Diastolic blood pressure (mmHg)",
    "LBXTC": "Total cholesterol (mg/dL)",
    "LBXGLU": "Fasting blood glucose (mg/dL)",
    "LBXGH": "Glycohemoglobin (HbA1c %)",
    "LBXHCT": "Hematocrit (%)",
    "LBXWBCSI": "White blood cell count (1000/uL)",
    "LBXPLTSI": "Platelet count (1000/uL)",
    "MCQ010": "Ever told had asthma (0=No,1=Yes)",
    "MCQ160A": "Ever told had arthritis",
    "MCQ160B": "Ever told had heart failure",
    "MCQ160C": "Ever told had coronary heart disease",
    "MCQ160D": "Ever told had angina",
    "MCQ160E": "Ever told had heart attack",
    "MCQ160F": "Ever told had stroke",
    "MCQ160G": "Ever told had emphysema",
    "MCQ160K": "Ever told had cancer/malignancy",
    "PAD615": "Minutes of moderate physical activity/week",
    "PAD630": "Minutes of vigorous physical activity/week",
    "PAQ605": "Walking/biking for transportation (Yes/No)",
    "SMQ020": "Current smoking status",
    "SMQ040": "Ever smoked at least 100 cigarettes",
    "ALQ101": "Had at least 12 alcohol drinks lifetime",
    "ALQ120Q": "Days drank alcohol in past 12 months",
    "DR1TKCAL": "Daily total calorie intake",
    "DR1TPROT": "Daily protein intake (g)",
    "DR1TCARB": "Daily carbohydrate intake (g)",
    "DR1TSUGR": "Daily total sugar intake (g)",
    "DR1TTFAT": "Daily total fat intake (g)",
    "DR1TSODI": "Daily sodium intake (mg)",
    "RXDUSE": "Currently taking prescription meds"
    }

fields = {}
for name in feature_names:
    default = feature_means.get(name, np.nan)
    desc = feature_descriptions.get(name, name)

    fields[name] = (
        float,
        Field(default=float(default) if default == default else None,
              description=desc)
    )

InsomniaInput = create_model("InsomniaInput", **fields)

@app.get("/")
def root():
    return {"status": "Insomnia Prediction API is running"}

@app.get("/features")
def list_features():
    return {
        "features": [
            {
                "name": f,
                "default": feature_means.get(f),
                "description": feature_descriptions.get(f, "")
            }
            for f in feature_names
        ]
    }

import numpy as np

def sanitize_for_json(d):
    clean = {}
    for k, v in d.items():
        if v is None:
            clean[k] = None
        elif isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                clean[k] = None
            else:
                clean[k] = float(v)
        else:
            clean[k] = v
    return clean

@app.post("/predict")
def predict(payload: InsomniaInput):

    row = {
        f: getattr(payload, f, feature_means.get(f))
        for f in feature_names
    }
    df = pd.DataFrame([row], columns=feature_names)
    input_dict = payload.dict()

    all_mean_input = True
    for name in feature_names:
        try:
            if float(input_dict.get(name, feature_means[name])) != float(feature_means[name]):
                all_mean_input = False
                break
        except Exception:
            all_mean_input = False
            break

    guidance_note = (
        "Warning: All values are defaults averages! "
        "This can produce very high insomnia probabilities because the model "
        "is optimized for recall!"
        if all_mean_input
        else
            "Remember this is just what the machine learning model believes, so please take this into light consideration! "

    )
    proba = float(model.predict_proba(df)[0, 1])
    label = int(proba >= best_threshold)

    def risk_band(proba: float) -> str:
      if proba < 0.30:
        return "low"
      elif proba < 0.60:
        return "moderate"
      else:
        return "high"

    base_pipeline = model.estimator
    X_trans = base_pipeline.named_steps["imputer"].transform(df)
    X_trans = base_pipeline.named_steps["scaler"].transform(X_trans)

    shap_vals = explainer.shap_values(X_trans)[0]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals,
        "abs_shap": np.abs(shap_vals)
    })

    top_features = (
        shap_df.sort_values("abs_shap", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )
    risk = risk_band(proba)

    return {
      "insomnia_probability": float(proba),
      "predicted_label": int(proba >= best_threshold),
      "threshold_used": best_threshold,
      "risk_band": risk,
      "top_shap_features": top_features
}

    