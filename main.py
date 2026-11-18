
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ==========================================
# 1. LOAD MODEL AND PREPROCESSING INFORMATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

try:
    model = joblib.load("model/accident_severity_rf_compressed.joblib")

    preprocessing_info = joblib.load(
        MODEL_DIR / "preprocessing_info.joblib"
    )

except Exception as e:
    raise RuntimeError(
        f"Unable to load model or preprocessing files: {e}"
    )


# Extract preprocessing information
feature_columns = preprocessing_info["feature_columns"]

top_weather = preprocessing_info["top_weather"]
top_states = preprocessing_info["top_states"]

numeric_weather_cols = preprocessing_info["numeric_weather_cols"]
road_features = preprocessing_info["road_features"]
twilight_cols = preprocessing_info["twilight_cols"]

numeric_medians = preprocessing_info["numeric_medians"]


# ==========================================
# 2. CUSTOM SEVERITY MESSAGES
# ==========================================

severity_messages = {
    1: (
        "Minor accident - Low predicted severity. "
        "Follow normal safety and reporting procedures."
    ),

    2: (
        "Moderate accident - Exercise caution and "
        "assess the situation carefully."
    ),

    3: (
        "Serious accident - Seek emergency assistance "
        "and medical evaluation."
    ),

    4: (
        "Severe accident - Contact emergency services "
        "immediately."
    )
}


# ==========================================
# 3. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Accident Severity Prediction API",
    description=(
        "An API that predicts accident severity "
        "using a trained Random Forest model."
    ),
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ==========================================
# 4. BOOLEAN CONVERSION FUNCTION
# ==========================================

def convert_boolean(value):

    if pd.isna(value):
        return 0

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, str):

        value = value.strip().lower()

        if value in ["true", "yes", "1"]:
            return 1

        elif value in ["false", "no", "0", ""]:
            return 0

        else:
            raise ValueError(
                f"Invalid boolean value: {value}"
            )

    return int(bool(value))


# ==========================================
# 5. PREPROCESS INPUT DATA
# ==========================================

def preprocess_accident(data):

    df = pd.DataFrame([data])

    # Check required columns
    required_columns = [
        "Start_Time",
        "Start_Lat",
        "Start_Lng"
    ]

    for col in required_columns:

        if col not in df.columns or pd.isna(df[col].iloc[0]):
            raise ValueError(
                f"Missing required input: {col}"
            )

    # --------------------------------------
    # Timestamp processing
    # --------------------------------------

    timestamp = pd.to_datetime(
        df["Start_Time"].iloc[0],
        format="mixed",
        errors="coerce"
    )

    if pd.isna(timestamp):
        raise ValueError(
            "Invalid Start_Time format."
        )

    df["Accident_Hour"] = timestamp.hour
    df["Accident_Day"] = timestamp.dayofweek
    df["Accident_Month"] = timestamp.month
    df["Accident_Year"] = timestamp.year

    # Weekend indicator
    df["Is_Weekend"] = int(
        timestamp.dayofweek in [5, 6]
    )

    # Rush hour indicator
    df["Is_Rush_Hour"] = int(
        timestamp.hour in [7, 8, 9, 16, 17, 18, 19]
    )

    # --------------------------------------
    # Numeric weather feature processing
    # --------------------------------------

    for col in numeric_weather_cols:

        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if col == "Precipitation(in)":
            df[col] = df[col].fillna(0)

        elif col == "Wind_Chill(F)":

            if "Temperature(F)" in df.columns:
                df[col] = df[col].fillna(
                    df["Temperature(F)"]
                )

            df[col] = df[col].fillna(
                numeric_medians.get(col, 0)
            )

        else:
            df[col] = df[col].fillna(
                numeric_medians.get(col, 0)
            )

    # --------------------------------------
    # Road feature processing
    # --------------------------------------

    for col in road_features:

        if col not in df.columns:
            df[col] = 0

        df[col] = df[col].apply(
            convert_boolean
        )

    # --------------------------------------
    # Twilight feature processing
    # --------------------------------------

    for col in twilight_cols:

        if col not in df.columns:
            df[col] = "DAY"

        df[col] = (
            df[col]
            .fillna("DAY")
            .astype(str)
            .str.strip()
            .str.upper()
            .apply(lambda x: 1 if x == "NIGHT" else 0)
        )

    # --------------------------------------
    # Weather condition processing
    # --------------------------------------

    weather = str(
        data.get("Weather_Condition", "Other")
    ).strip()

    if weather not in top_weather:
        weather = "Other"

    # --------------------------------------
    # State processing
    # --------------------------------------

    state = str(
        data.get("State", "Other")
    ).strip().upper()

    if state not in top_states:
        state = "Other"

    # --------------------------------------
    # Construct final feature dataframe
    # --------------------------------------

    input_features = {}

    for col in feature_columns:

        if col.startswith("Weather_Condition_"):

            category = col.replace(
                "Weather_Condition_", ""
            )

            input_features[col] = int(
                category == weather
            )

        elif col.startswith("State_Grouped_"):

            category = col.replace(
                "State_Grouped_", ""
            )

            input_features[col] = int(
                category == state
            )

        elif col in df.columns:

            input_features[col] = df[col].iloc[0]

        else:

            input_features[col] = numeric_medians.get(
                col, 0
            )

    # Create dataframe in exact training feature order
    input_df = pd.DataFrame(
        [input_features],
        columns=feature_columns
    )

    input_df = input_df.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    # Validate feature count
    if input_df.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: expected "
            f"{model.n_features_in_}, got {input_df.shape[1]}"
        )

    return input_df


# ==========================================
# 6. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "status": "Running",
        "project": "Accident Severity Prediction",
        "model": "Random Forest Classifier",
        "features": len(feature_columns),
        "message": "API is ready for predictions."
    }


# ==========================================
# 7. ACCIDENT SEVERITY PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(data: dict):

    try:

        # Preprocess incoming data
        input_df = preprocess_accident(data)

        # Generate prediction
        prediction = model.predict(input_df)[0]

        severity = int(prediction)

        # Generate class probabilities
        probabilities = model.predict_proba(input_df)[0]

        # Get custom severity message
        severity_message = severity_messages.get(
            severity,
            "Unknown accident severity."
        )

        # Prepare probability output
        class_probabilities = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(
                model.classes_,
                probabilities
            )
        }

        # Return API response
        return {
            "predicted_severity": severity,
            "severity_message": severity_message,
            "class_probabilities": class_probabilities,
            "message": "Accident severity predicted successfully"
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


