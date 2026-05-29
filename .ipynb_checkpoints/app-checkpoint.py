# =========================================================
# STREAMLIT MAIN APP
# =========================================================

import streamlit as st
import pandas as pd
from utils.prediction_utils import predict_all

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Malnutrition Prediction System",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("Child Malnutrition Prediction System (Ethiopia)")

st.markdown("""
### Predicting Undernutrition Among Under-Five Children in Ethiopia Using Machine Learning

This system predicts:
- Stunting
- Wasting
- Underweight
""")

# =========================================================
# INPUT SECTION
# =========================================================

st.header("Child Information")

# =========================================================
# CHILD CHARACTERISTICS
# =========================================================

with st.expander("Child Characteristics", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        age_in_month = st.number_input("Age (Months)", 0, 59, 24)

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

        height = st.number_input("Height (cm)", 30.0, 150.0, 85.0)
        weight = st.number_input("Weight (kg)", 2.0, 30.0, 10.0)

    with col2:
        ever_bf_binary = st.selectbox("Ever Breastfed", [0, 1])
        ebf_months = st.number_input("Exclusive Breastfeeding (Months)", 0.0, 24.0, 6.0)
        vaccine_miss = st.selectbox("Missed Vaccination", [0, 1])
        suffered_illness = st.selectbox("Recent Illness", [0, 1])

    with col3:
        num_childrenU5 = st.number_input("Children Under Five", 0, 10, 2)
        hh_members = st.number_input("Household Members", 1, 20, 5)

        residence = st.selectbox(
            "Residence",
            [0, 1],
            format_func=lambda x: "Rural" if x == 0 else "Urban"
        )

# =========================================================
# MATERNAL & HOUSEHOLD
# =========================================================

with st.expander("Maternal & Household Information", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        mother_age = st.number_input("Mother Age", 15, 50, 28)
        mother_age_marriage = st.number_input("Mother Age at Marriage", 10, 40, 18)
        antenatal_checkups = st.number_input("Antenatal Checkups", 0, 20, 4)

        skilled_birth_assist = st.selectbox("Skilled Birth Assistance", [0, 1])

    with col2:
        educ_cat = st.selectbox("Maternal Education", [1, 2, 3, 4, 5])
        media_access = st.selectbox("Media Access", [0, 1])
        own_business = st.selectbox("Own Business", [0, 1])
        HHhead_male = st.selectbox("Household Head Sex", [0, 1])

    with col3:
        HHhead_occup = st.selectbox("Household Head Occupation", [1,2,3,4,5,6,7,8,9])

        asset_index = st.selectbox(
            "Wealth Index",
            [1,2,3,4,5],
            format_func=lambda x: ["Poorest","Poorer","Middle","Rich","Richest"][x-1]
        )

        FS_category = st.selectbox(
            "Food Security",
            [1,2,3,4],
            format_func=lambda x: [
                "Food Secure",
                "Mildly Insecure",
                "Moderately Insecure",
                "Severely Insecure"
            ][x-1]
        )

# =========================================================
# ENVIRONMENT
# =========================================================

with st.expander("Environment & Health Access", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        region = st.selectbox("Region", [1,2,3,4,5,6,7,8,12,13,14,15])
        dist_hf = st.number_input("Distance to Health Facility (Hours)", 0.0, 24.0, 1.0)
        covered_health_insurance = st.selectbox("Health Insurance", [0, 1])

    with col2:
        dr_water_im_um = st.selectbox("Drinking Water Source", [0, 1])
        toilet_im_um = st.selectbox("Toilet Facility", [0, 1])
        moth_ors_knowledge = st.selectbox("Mother ORS Knowledge", [0, 1])

    with col3:
        mother_live_hh = st.selectbox("Mother Lives in Household", [0, 1])
        father_live_hh = st.selectbox("Father Lives in Household", [0, 1])

# =========================================================
# INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame([{
    "age_in_month": age_in_month,
    "sex": sex,
    "height": height,
    "weight": weight,
    "residence": residence,
    "region": region,
    "hh_members": hh_members,
    "num_childrenU5": num_childrenU5,
    "mother_age": mother_age,
    "mother_age_marriage": mother_age_marriage,
    "antenatal_checkups": antenatal_checkups,
    "ebf_months": ebf_months,
    "dist_hf": dist_hf,
    "asset_index": asset_index,
    "FS_category": FS_category,
    "skilled_birth_assist": skilled_birth_assist,
    "ever_bf_binary": ever_bf_binary,
    "vaccine_miss": vaccine_miss,
    "suffered_illness": suffered_illness,
    "educ_cat": educ_cat,
    "dr_water_im_um": dr_water_im_um,
    "toilet_im_um": toilet_im_um,
    "own_business": own_business,
    "covered_health_insurance": covered_health_insurance,
    "mother_live_hh": mother_live_hh,
    "father_live_hh": father_live_hh,
    "media_access": media_access,
    "moth_ors_knowledge": moth_ors_knowledge,
    "HHhead_male": HHhead_male,
    "HHhead_occup": HHhead_occup
}])

# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Malnutrition Risk"):

    result = predict_all(input_data)

    st.subheader("Prediction Results")

    # =====================================================
    # HELPER FUNCTION
    # =====================================================

    def show_result(title, probability, risk):

        probability = float(probability)

        st.markdown(f"## {title}")

        st.metric("Risk Probability", f"{probability:.2f}%")

        st.progress(probability / 100.0)

        if risk == "HIGH RISK":
            st.error(risk)
        else:
            st.success(risk)

        st.write(f"Predicted Probability: {probability:.2f}%")

        # Interpretation
        if probability >= 80:
            st.warning("Very high risk")
        elif probability >= 60:
            st.info("Moderate risk")
        elif probability >= 40:
            st.info("Borderline risk")
        else:
            st.success("Low risk")

    # =====================================================
    # RESULTS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        show_result(
            "Stunting",
            result["Stunting"]["probability"],
            result["Stunting"]["risk"]
        )

    with col2:
        show_result(
            "Wasting",
            result["Wasting"]["probability"],
            result["Wasting"]["risk"]
        )

    with col3:
        show_result(
            "Underweight",
            result["Underweight"]["probability"],
            result["Underweight"]["risk"]
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    st.markdown("---")
    st.subheader("Clinical Summary")

    high_risk = []

    if result["Stunting"]["risk"] == "HIGH RISK":
        high_risk.append("Stunting")

    if result["Wasting"]["risk"] == "HIGH RISK":
        high_risk.append("Wasting")

    if result["Underweight"]["risk"] == "HIGH RISK":
        high_risk.append("Underweight")

    if len(high_risk) == 0:
        st.success("No high-risk conditions detected.")
    else:
        st.warning("High risk detected in: " + ", ".join(high_risk))
        st.info("Recommend clinical follow-up.")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
### Research Information

**Title:** Predicting Undernutrition Among Under-Five Children in Ethiopia Using Machine Learning

**Researcher:** Telila Kejela (MSc, Addis Ababa University)

**Supervisors:**
- Dr. Getachew H/Mariam  
- Prof. Abera Kumie
""")