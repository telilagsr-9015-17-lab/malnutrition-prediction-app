# =========================================================
# IMPORTS
# =========================================================

import os
import joblib
import numpy as np
import pandas as pd

# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

# =========================================================
# LOAD FUNCTION
# =========================================================

def load_model(file_name):

    return joblib.load(
        os.path.join(MODEL_DIR, file_name)
    )

# =========================================================
# LOAD MODELS
# =========================================================

gbm_stunting = load_model(
    "gbm_stunting.pkl"
)

xgb_wasting = load_model(
    "xgb_wasting.pkl"
)

svm_underweight = load_model(
    "svm_underweight.pkl"
)

# =========================================================
# LOAD PREPROCESSORS
# =========================================================

preprocessor_stunting = load_model(
    "preprocessor_stunting.pkl"
)

preprocessor_wasting = load_model(
    "preprocessor_wasting.pkl"
)

preprocessor_underweight = load_model(
    "preprocessor_underweight.pkl"
)

# =========================================================
# LOAD FEATURE INDEXES
# =========================================================

selected_idx_stunting = load_model(
    "selected_idx_stunting.pkl"
)

selected_idx_wasting = load_model(
    "selected_idx_wasting.pkl"
)

selected_idx_underweight = load_model(
    "selected_idx_underweight.pkl"
)

# =========================================================
# LOAD THRESHOLDS
# =========================================================

thresholds = load_model(
    "thresholds.pkl"
)

# =========================================================
# TO DENSE
# =========================================================

def to_dense(X):

    if hasattr(X, "toarray"):

        return X.toarray()

    return X

# =========================================================
# PREPROCESS FUNCTION
# =========================================================

def preprocess_input(input_df):

    # =====================================================
    # COPY INPUT
    # =====================================================

    input_df = input_df.copy()

    # =====================================================
    # CATEGORICAL FEATURES
    # =====================================================

    categorical_columns = [

        "region",
        "asset_index",
        "residence",
        "sex",
        "skilled_birth_assist",
        "ever_bf_binary",
        "vaccine_miss",
        "suffered_illness",
        "educ_cat",
        "FS_category",
        "dr_water_im_um",
        "toilet_im_um",
        "own_business",
        "covered_health_insurance",
        "mother_live_hh",
        "father_live_hh",
        "media_access",
        "moth_ors_knowledge",
        "HHhead_male",
        "HHhead_occup"
    ]

    # =====================================================
    # NUMERICAL FEATURES
    # =====================================================

    numerical_columns = [

        "antenatal_checkups",
        "ebf_months",
        "mother_age",
        "dist_hf",
        "num_childrenU5",
        "age_in_month",
        "hh_members",
        "mother_age_marriage",
        "height",
        "weight"
    ]

    # =====================================================
    # FORCE TRAINING TYPES
    # =====================================================

    # IMPORTANT:
    # categorical -> STRING
    # numerical -> FLOAT

    for col in categorical_columns:

        input_df[col] = (
            pd.to_numeric(
                input_df[col],
                errors="coerce"
            )
            .astype(int)
            .astype(str)
        )

    for col in numerical_columns:

        input_df[col] = pd.to_numeric(
            input_df[col],
            errors="coerce"
        ).astype(float)

    # =====================================================
    # STUNTING
    # =====================================================

    X_stunting = preprocessor_stunting.transform(
        input_df
    )

    if hasattr(X_stunting, "toarray"):

        X_stunting = X_stunting.toarray()

    X_stunting = X_stunting[
        :,
        selected_idx_stunting
    ]

    X_stunting = np.array(
        X_stunting,
        dtype=np.float32
    )

    # =====================================================
    # WASTING
    # =====================================================

    X_wasting = preprocessor_wasting.transform(
        input_df
    )

    if hasattr(X_wasting, "toarray"):

        X_wasting = X_wasting.toarray()

    X_wasting = X_wasting[
        :,
        selected_idx_wasting
    ]

    X_wasting = np.array(
        X_wasting,
        dtype=np.float32
    )

    # =====================================================
    # UNDERWEIGHT
    # =====================================================

    X_underweight = preprocessor_underweight.transform(
        input_df
    )

    if hasattr(X_underweight, "toarray"):

        X_underweight = X_underweight.toarray()

    X_underweight = X_underweight[
        :,
        selected_idx_underweight
    ]

    X_underweight = np.array(
        X_underweight,
        dtype=np.float32
    )

    return (

        X_stunting,

        X_wasting,

        X_underweight
    )

# =========================================================
# RISK LABEL
# =========================================================

def get_risk_label(prob, threshold):

    if prob >= threshold:

        return "HIGH RISK"

    return "LOW RISK"

# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================

def predict_all(input_df):

    (
        X_stunting,
        X_wasting,
        X_underweight

    ) = preprocess_input(input_df)

    # =====================================================
    # PREDICT PROBABILITIES
    # =====================================================

    stunting_prob = gbm_stunting.predict_proba(

        X_stunting
    )[0][1]

    wasting_prob = xgb_wasting.predict_proba(

        X_wasting
    )[0][1]

    underweight_prob = svm_underweight.predict_proba(

        X_underweight
    )[0][1]

    # =====================================================
    # APPLY THRESHOLDS
    # =====================================================

    stunting_risk = get_risk_label(

        stunting_prob,

        thresholds["stunting"]
    )

    wasting_risk = get_risk_label(

        wasting_prob,

        thresholds["wasting"]
    )

    underweight_risk = get_risk_label(

        underweight_prob,

        thresholds["underweight"]
    )

    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "Stunting": {

            "probability":
            round(stunting_prob * 100, 2),

            "risk":
            stunting_risk
        },

        "Wasting": {

            "probability":
            round(wasting_prob * 100, 2),

            "risk":
            wasting_risk
        },

        "Underweight": {

            "probability":
            round(underweight_prob * 100, 2),

            "risk":
            underweight_risk
        }
    }