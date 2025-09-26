import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "models", "titanic_survival_predictor_pipeline.pkl")
try:
    model_pipeline = joblib.load(model_path)
    st.success("Machine Learning Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}. Please ensure it is saved correctly.")
    st.info("You might need to run the 'explore.ipynb' notebook to train and save the model first.")
    st.stop()
except Exception as e:
    st.error(f"An error occured while loading the model: {e}")
    st.stop()
st.title("Titanic Survival Predictor")
st.markdown("---")
st.write("Enter passenger details to predict their survival on the Titanic.")
st.header("Passenger Information")
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], format_func = lambda x: f"Class {x}")
    sex = st.radio("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 30)
    sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 6, 0)
with col2:
    parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.number_input("Fare ($)", min_value = 0.0, max_value = 1000.0, value = 30.0, step = 1.0)
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"], format_func = lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x])
if st.button("Predict Survival"):
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked

    }])
    st.write("---")
    st.subheader("Prediction Result:")
    try:
        prediction = model_pipeline.predict(input_data)[0]
        prediction_proba = model_pipeline.predict_proba(input_data)[0]
        if prediction == 1:
            st.success(f"**Survived!** 🎉 (Confidence: {prediction_proba[1]*100:.2f}%)")
            st.balloons()
        else:
            st.error(f"**Did Not Survive.** 😔 (Confidence: {prediction_proba[0]*100:.2f}%)")
        st.write("---")
        st.subheader("Input Data Summary:")
        st.dataframe(input_data)
    except Exception as e:
        st.error(f"An error occured during prediction: {e}")
        st.info("Please check your input values.")
st.markdown("---")
st.caption("Developed as part of the 4Geeks Academy ML Web App Tutorial.")