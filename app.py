import streamlit as st
import numpy as np
import pickle

# load model
model, scaler = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="HealSync Lite", page_icon="💙")

st.title("💙 HealSync Lite — Burnout Risk Predictor")
st.write("A simple AI tool that helps you understand early burnout patterns.")

st.subheader("Daily Wellness Check-In")

sleep = st.slider("How many hours did you sleep yesterday?", 0, 12, 7)
stress = st.slider("How stressed do you feel today? (1 = calm, 10 = overwhelmed)", 1, 10, 5)
mood = st.slider("How is your mood? (1 = low, 5 = great)", 1, 5, 3)
screen = st.slider("How many hours of screen time did you have?", 0, 12, 4)

if st.button("Check Burnout Risk"):
    features = scaler.transform([[sleep, stress, mood, screen]])
    prediction = model.predict(features)[0]

    st.subheader("🧠 Burnout Risk Result")

    if prediction == 1:
        st.error("🔴 **High Burnout Risk Detected**")
        st.write("Your recent patterns indicate elevated stress and fatigue.")
        st.write("**Suggested Action:** Take a short break, rest well tonight, and check in with yourself tomorrow.")
    else:
        st.success("🟢 **Low Burnout Risk**")
        st.write("You're doing well today — keep maintaining your balance!")

st.write("---")
st.caption("HealSync Lite — Prototype wellness tool using machine learning (2025)")
