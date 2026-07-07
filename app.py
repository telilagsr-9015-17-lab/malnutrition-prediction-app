# ==========================================================
# CHILD MALNUTRITION PREDICTION SYSTEM
# Ethiopia
#
# Research Title:
# Predicting Undernutrition Among Under-Five Children
# in Ethiopia Using Machine Learning
#
# Researcher:
# Telila Kejela
#
# ==========================================================

import streamlit as st
import pandas as pd

from utils.prediction_utils import predict_all

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Malnutrition Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

div[data-testid="metric-container"]{
    border-radius:12px;
    padding:18px;
    border:1px solid #E6E6E6;
    background:#FAFAFA;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:20px;
    font-weight:bold;
    border-radius:10px;
}

hr{
    margin-top:35px;
    margin-bottom:35px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LABEL DICTIONARIES
# ==========================================================

YES_NO = {
    1: "Yes",
    0: "No"
}

GENDER = {
    0: "Female",
    1: "Male"
}

RESIDENCE = {
    0: "Rural",
    1: "Urban"
}

WATER = {
    0: "Unimproved",
    1: "Improved"
}

TOILET = {
    0: "Unimproved",
    1: "Improved"
}

REGIONS = {
    1: "Tigray",
    2: "Afar",
    3: "Amhara",
    4: "Oromia",
    5: "Somali",
    6: "Benishangul-Gumuz",
    7: "SNNPR",
    8: "Sidama",
    12: "Gambella",
    13: "Harari",
    14: "Addis Ababa",
    15: "Dire Dawa"
}

EDUCATION = {
    1: "No education",
    2: "Informal",
    3: "Primary",
    4: "Secondary",
    5: "Higher"
}

WEALTH = {
    1: "Poorest",
    2: "Poorer",
    3: "Middle",
    4: "Rich",
    5: "Richest"
}

FOOD_SECURITY = {
    1: "Food Secure",
    2: "Mildly Insecure",
    3: "Moderately Insecure",
    4: "Severely Insecure"
}

OCCUPATION = {

    -777: "Other",

    1: "Student",

    2: "Professional / Technical / Managerial",

    3: "Clerical",

    4: "Sales and Services",

    5: "Skilled Manual",

    6: "Unskilled Manual",

    7: "Agriculture",

    8: "Domestic Service",

    9: "Unemployed"

}

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def yes_no(label, help_text=""):
    """
    Binary Yes / No selector
    Returns:
        Yes -> 1
        No  -> 0
    """
    return st.selectbox(
        label,
        options=[1, 0],
        format_func=lambda x: YES_NO[x],
        help=help_text
    )


def male_female(label, help_text=""):
    """
    Gender selector
    Returns:
        Female -> 0
        Male   -> 1
    """
    return st.selectbox(
        label,
        options=[0, 1],
        format_func=lambda x: GENDER[x],
        help=help_text
    )


def residence_box():

    return st.selectbox(
        "Residence",
        options=list(RESIDENCE.keys()),
        format_func=lambda x: RESIDENCE[x],
        help="Place where the child lives."
    )


def region_box():

    return st.selectbox(
        "Region",
        options=list(REGIONS.keys()),
        format_func=lambda x: REGIONS[x],
        help="Administrative region of residence."
    )


def education_box():

    return st.selectbox(
        "Maternal Education",
        options=list(EDUCATION.keys()),
        format_func=lambda x: EDUCATION[x],
        help="Highest education attained by the mother."
    )


def wealth_box():

    return st.selectbox(
        "Household Wealth Index",
        options=list(WEALTH.keys()),
        format_func=lambda x: WEALTH[x],
        help="Economic status of the household."
    )


def food_security_box():

    return st.selectbox(
        "Household Food Security",
        options=list(FOOD_SECURITY.keys()),
        format_func=lambda x: FOOD_SECURITY[x],
        help="Food security classification."
    )


def occupation_box():

    return st.selectbox(
        "Household Head Occupation",
        options=[
            -777,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9
        ],
        format_func=lambda x: OCCUPATION[x],
        help="Main occupation of the household head."
    )


def improved_water():

    return st.selectbox(
        "Drinking Water Source",
        options=[1, 0],
        format_func=lambda x: WATER[x],
        help="Improved or unimproved drinking water source."
    )


def improved_toilet():

    return st.selectbox(
        "Toilet Facility",
        options=[1, 0],
        format_func=lambda x: TOILET[x],
        help="Improved or unimproved sanitation facility."
    )


# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("🩺 Child Malnutrition Prediction System")

st.markdown("""
### Predicting Undernutrition Among Under-Five Children in Ethiopia Using Machine Learning

This application predicts the probability of:

- **Stunting**
- **Wasting**
- **Underweight**

using machine learning models developed from Ethiopian child health data.

---

Please complete all sections below before running the prediction.
""")

# ==========================================================
# INPUT SECTION HEADER
# ==========================================================

st.header("📋 Child Information")

# ==========================================================
# CHILD CHARACTERISTICS
# ==========================================================

with st.expander(
    "👶 Child Characteristics",
    expanded=True
):

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------
    # COLUMN 1
    # ------------------------------------------------------

    with col1:

        age_in_month = st.number_input(
            "Age (Months)",
            min_value=6,
            max_value=59,
            value=24,
            step=1,
            help="Age of the child in completed months (6–59 months)."
        )

        sex = male_female(
            "Sex",
            help_text="Select the biological sex of the child."
        )

        height = st.number_input(
            "Height (cm)",
            min_value=30.0,
            max_value=150.0,
            value=85.0,
            step=0.1,
            help="Measured height or length of the child in centimeters."
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=2.0,
            max_value=30.0,
            value=10.0,
            step=0.1,
            help="Measured body weight of the child in kilograms."
        )

    # ------------------------------------------------------
    # COLUMN 2
    # ------------------------------------------------------

    with col2:

        ever_bf_binary = yes_no(
            "Ever Breastfed",
            help_text="Has the child ever been breastfed?"
        )

        ebf_months = st.number_input(
            "Exclusive Breastfeeding Duration (Months)",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5,
            help="Number of months the child was exclusively breastfed."
        )

        vaccine_miss = yes_no(
            "Missed Vaccination",
            help_text="Has the child missed any scheduled vaccination?"
        )

        suffered_illness = yes_no(
            "Recent Illness",
            help_text="Has the child experienced illness recently?"
        )

    # ------------------------------------------------------
    # COLUMN 3
    # ------------------------------------------------------

    with col3:

        num_childrenU5 = st.number_input(
            "Children Under Five in Household",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Number of children younger than five years living in the household."
        )

        hh_members = st.number_input(
            "Household Members",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Total number of people living in the household."
        )

        residence = residence_box()

# ==========================================================
# MATERNAL & HOUSEHOLD INFORMATION
# ==========================================================

with st.expander(
    "👩 Maternal & Household Information",
    expanded=True
):

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------
    # COLUMN 1
    # ------------------------------------------------------

    with col1:

        mother_age = st.number_input(
            "Mother's Current Age (Years)",
            min_value=15,
            max_value=50,
            value=28,
            step=1,
            help="Current age of the child's mother."
        )

        mother_age_marriage = st.number_input(
            "Mother's Age at Marriage",
            min_value=10,
            max_value=40,
            value=18,
            step=1,
            help="Age of the mother when she first married."
        )

        antenatal_checkups = st.number_input(
            "Number of Antenatal Care Visits",
            min_value=0,
            max_value=20,
            value=4,
            step=1,
            help="Total ANC visits during pregnancy."
        )

        skilled_birth_assist = yes_no(
            "Skilled Birth Assistance",
            help_text="Was the delivery assisted by a skilled health professional?"
        )

    # ------------------------------------------------------
    # COLUMN 2
    # ------------------------------------------------------

    with col2:

        educ_cat = education_box()

        media_access = yes_no(
            "Media Access",
            help_text="Does the household have access to radio, TV, newspapers, or similar media?"
        )

        own_business = yes_no(
            "Household Owns a Business",
            help_text="Does the household own or operate a business?"
        )

        HHhead_male = male_female(
            "Sex of Household Head",
            help_text="Select the sex of the household head."
        )

    # ------------------------------------------------------
    # COLUMN 3
    # ------------------------------------------------------

    with col3:

        HHhead_occup = occupation_box()

        asset_index = wealth_box()

        FS_category = food_security_box()


# ==========================================================
# ENVIRONMENT & HEALTH ACCESS
# ==========================================================

with st.expander(
    "🏥 Environment & Health Access",
    expanded=True
):

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------
    # COLUMN 1
    # ------------------------------------------------------

    with col1:

        region = region_box()

        dist_hf = st.number_input(
            "Distance to Health Facility (Hours)",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.5,
            help="Approximate travel time from the household to the nearest health facility."
        )

        covered_health_insurance = yes_no(
            "Covered by Health Insurance",
            help_text="Is the household covered by any health insurance scheme?"
        )

    # ------------------------------------------------------
    # COLUMN 2
    # ------------------------------------------------------

    with col2:

        dr_water_im_um = improved_water()

        toilet_im_um = improved_toilet()

        moth_ors_knowledge = yes_no(
            "Mother Has ORS Knowledge",
            help_text="Does the mother know about Oral Rehydration Solution (ORS)?"
        )

    # ------------------------------------------------------
    # COLUMN 3
    # ------------------------------------------------------

    with col3:

        mother_live_hh = yes_no(
            "Mother Lives in Household",
            help_text="Does the child's mother currently live in the household?"
        )

        father_live_hh = yes_no(
            "Father Lives in Household",
            help_text="Does the child's father currently live in the household?"
        )

# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

input_data = pd.DataFrame([{

    # --------------------------------------------------
    # CHILD
    # --------------------------------------------------

    "age_in_month": age_in_month,
    "sex": sex,
    "height": height,
    "weight": weight,

    # --------------------------------------------------
    # HOUSEHOLD
    # --------------------------------------------------

    "residence": residence,
    "region": region,
    "hh_members": hh_members,
    "num_childrenU5": num_childrenU5,

    # --------------------------------------------------
    # MATERNAL
    # --------------------------------------------------

    "mother_age": mother_age,
    "mother_age_marriage": mother_age_marriage,
    "antenatal_checkups": antenatal_checkups,
    "ebf_months": ebf_months,
    "dist_hf": dist_hf,

    # --------------------------------------------------
    # SOCIOECONOMIC
    # --------------------------------------------------

    "asset_index": asset_index,
    "FS_category": FS_category,

    # --------------------------------------------------
    # BINARY VARIABLES
    # --------------------------------------------------

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

# ==========================================================
# OPTIONAL: DISPLAY INPUT SUMMARY
# ==========================================================

with st.expander("📄 Review Entered Information", expanded=False):

    st.info(
        "Review the entered values below before running the prediction."
    )

    preview = input_data.copy()

    preview["sex"] = preview["sex"].map(GENDER)
    preview["residence"] = preview["residence"].map(RESIDENCE)
    preview["region"] = preview["region"].map(REGIONS)
    preview["educ_cat"] = preview["educ_cat"].map(EDUCATION)
    preview["asset_index"] = preview["asset_index"].map(WEALTH)
    preview["FS_category"] = preview["FS_category"].map(FOOD_SECURITY)
    preview["HHhead_occup"] = preview["HHhead_occup"].map(OCCUPATION)

    binary_columns = [
        "skilled_birth_assist",
        "ever_bf_binary",
        "vaccine_miss",
        "suffered_illness",
        "dr_water_im_um",
        "toilet_im_um",
        "own_business",
        "covered_health_insurance",
        "mother_live_hh",
        "father_live_hh",
        "media_access",
        "moth_ors_knowledge"
    ]

    for col in binary_columns:
        if col in ["dr_water_im_um", "toilet_im_um"]:
            mapping = WATER if col == "dr_water_im_um" else TOILET
            preview[col] = preview[col].map(mapping)
        else:
            preview[col] = preview[col].map(YES_NO)

    preview["HHhead_male"] = preview["HHhead_male"].map(GENDER)

    st.dataframe(
        preview.T,
        use_container_width=True
    )

st.markdown("---")

# ==========================================================
# PREDICTION SECTION
# ==========================================================

st.header("🩺 Prediction")

st.markdown(
    """
Click **Predict Malnutrition Risk** to estimate the probability of:

- Stunting
- Wasting
- Underweight
"""
)

predict_btn = st.button(
    "🔍 Predict Malnutrition Risk",
    use_container_width=True,
    type="primary"
)

# ==========================================================
# RESULT CARD
# ==========================================================

def show_result(title, probability, risk):

    probability = float(probability)

    if probability < 0:
        probability = 0

    if probability > 100:
        probability = 100

    st.markdown(f"## {title}")

    st.metric(
        label="Risk Probability",
        value=f"{probability:.2f}%"
    )

    st.progress(probability / 100)

    if risk == "HIGH RISK":
        st.error("🔴 HIGH RISK")

    else:
        st.success("🟢 LOW RISK")

    st.write(f"Predicted Probability: **{probability:.2f}%**")

    if probability >= 80:

        st.warning(
            "Very High Risk\n\nImmediate nutritional assessment and clinical follow-up are recommended."
        )

    elif probability >= 60:

        st.info(
            "Moderate Risk\n\nChild should receive nutritional monitoring and counselling."
        )

    elif probability >= 40:

        st.info(
            "Borderline Risk\n\nRoutine growth monitoring is advised."
        )

    else:

        st.success(
            "Low Risk\n\nContinue routine child healthcare and nutrition practices."
        )


# ==========================================================
# RUN PREDICTION
# ==========================================================

if predict_btn:

    with st.spinner("Running machine learning models..."):

        result = predict_all(input_data)

    st.markdown("---")

    st.header("📊 Prediction Results")

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

    # ======================================================
    # CLINICAL SUMMARY
    # ======================================================

    st.markdown("---")

    st.header("🩺 Clinical Summary")

    high_risk = []

    if result["Stunting"]["risk"] == "HIGH RISK":
        high_risk.append("Stunting")

    if result["Wasting"]["risk"] == "HIGH RISK":
        high_risk.append("Wasting")

    if result["Underweight"]["risk"] == "HIGH RISK":
        high_risk.append("Underweight")

    if len(high_risk) == 0:

        st.success(
            "✅ No high-risk malnutrition conditions were detected."
        )

    else:

        st.warning(
            "⚠ High-risk conditions detected: "
            + ", ".join(high_risk)
        )

        st.info(
            """
### Recommendation

The child is predicted to have an elevated risk of one or more forms of undernutrition.

Recommended actions include:

- Clinical examination

- Anthropometric assessment

- Nutritional counselling

- Follow-up by healthcare professionals

- Additional laboratory investigations when clinically indicated
"""
        )

    # ======================================================
    # PREDICTION SUMMARY TABLE
    # ======================================================

    st.markdown("---")

    st.subheader("📋 Prediction Summary")

    summary = pd.DataFrame({

        "Condition": [

            "Stunting",
            "Wasting",
            "Underweight"

        ],

        "Probability (%)": [

            round(float(result["Stunting"]["probability"]), 2),

            round(float(result["Wasting"]["probability"]), 2),

            round(float(result["Underweight"]["probability"]), 2)

        ],

        "Risk": [

            result["Stunting"]["risk"],

            result["Wasting"]["risk"],

            result["Underweight"]["risk"]

        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # DOWNLOAD RESULTS
    # ======================================================

    csv = summary.to_csv(index=False).encode("utf-8")

    st.download_button(

        "📥 Download Prediction Results",

        data=csv,

        file_name="malnutrition_prediction_results.csv",

        mime="text/csv",

        use_container_width=True

    )
# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/stethoscope.png",
        width=80
    )

    st.title("Malnutrition Prediction")

    st.markdown("---")

    st.markdown("""
### About

This application predicts the probability of:

- Stunting
- Wasting
- Underweight

among Ethiopian children aged **6–59 months** using machine learning models.

The prediction models remain unchanged from the research implementation.
""")

    st.markdown("---")

    st.markdown("""
### Data Requirements

Please ensure that:

- Child age is between **6–59 months**
- Height and weight are measured accurately
- Household information is entered correctly
- Maternal information is complete
""")

    st.markdown("---")

    st.success("Ready for prediction")

# ==========================================================
# RESEARCH INFORMATION
# ==========================================================

st.markdown("---")

st.markdown("""
## 📚 Research Information

**Title**

Predicting Undernutrition Among Under-Five Children in Ethiopia Using Machine Learning

**Researcher**

**Telila Kejela**

MSc Candidate

Addis Ababa University

---

### Supervisors

- Dr. Getachew H/Mariam
- Prof. Abera Kumie

---

### Disclaimer

This application is intended for **research and decision-support purposes only**.

Predictions generated by the machine learning models should **not replace professional medical diagnosis or clinical judgment**.

Children identified as high risk should be referred to qualified healthcare professionals for comprehensive assessment.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "© 2026 Telila Kejela | Child Malnutrition Prediction System | Addis Ababa University"
)

