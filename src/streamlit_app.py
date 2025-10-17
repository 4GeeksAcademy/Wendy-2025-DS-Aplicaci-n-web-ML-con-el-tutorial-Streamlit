import streamlit as st
import joblib
import pandas as pd
import os
DEFAULT_AGE = 30
DEFAULT_FARE = 30.0
MAX_FARE = 1000.0
MAX_RELATIVES = 6
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "models", "titanic_survival_predictor_pipeline.pkl")
try:
    model_pipeline = joblib.load(model_path)
    st.success(" ✅ Machine Learning Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌Error: Model file not found at `{model_path}`. Please ensure it is saved correctly.")
    st.info("You might need to run the training script to generate and save the model.")
    st.stop()
except Exception as e:
    st.error(f" ⚠️ Error loading the model: {e}")
    st.stop()
def predict_survival(model_pipeline, input_df):
    prediction = model_pipeline.predict(input_df)[0]
    prediction_proba = model_pipeline.predict_proba(input_df)[0]
    confidence = prediction_proba[prediction] * 100
    return prediction, confidence
st.title(" 🚢 Titanic Survival Predictor")
st.markdown("---")
st.write("Enter passenger details to predict survival on the Titanic.")
with st.form("Passenger Details"):
    st.header("Passenger Information")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func = lambda x: f"Class {x}")
        sex = st.radio("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, DEFAULT_AGE)
        sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, MAX_RELATIVES, 0)
    with col2:
        parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
        fare = st.number_input("Fare ($)", min_value = 0.0, max_value = MAX_FARE, value = DEFAULT_FARE, step = 1.0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], format_func = lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x])
    submitted = st.form_submit_button("Predict Survival")
if submitted:
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])
    st.markdown("---")
    st.subheader("🔍Prediction Result:")
    try:
        prediction, confidence = predict_survival(model_pipeline, input_data)
        if prediction == 1:
            st.success(f" 🎉 **Survived!** (Confidence: {confidence:.2f}%)")
            st.balloons()
        else:
            st.error(f" 😔 **Did Not Survive.** (Confidence: {confidence:.2f}%)")
        st.markdown("---")
        st.subheader("📋Input Data Summary:")
        st.dataframe(input_data)
    except Exception as e:
        st.error(f" ⚠️ Prediction error: {e}")
        st.info("Please check your input values.")
st.markdown("---")
st.caption(" 🛠  Wendy-2025-DS ML Web App Tutorial.")