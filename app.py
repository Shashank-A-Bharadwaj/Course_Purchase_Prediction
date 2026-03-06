import streamlit as st
import numpy as np
import pickle
import os

model = pickle.load(open(r"C:\Users\Dell\Downloads\internship\Course_Purchase_Prediction\model.pkl", "rb"))

st.title("EdTech Course Purchase Predictor")

st.write("Enter student details to predict course purchase.")

age = st.number_input("Age", 18, 60)
study_hours = st.number_input("Study Hours Per Week", 0, 40)
courses = st.number_input("Previous Courses Completed", 0, 20)
visits = st.number_input("Platform Visits Per Month", 0, 100)
assignment = st.number_input("Assignment Completion Rate (%)", 0, 100)

if st.button("Predict"):

    features = np.array([[age, study_hours, courses,visits, assignment]])

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    purchase_prob = probability[0][1] * 100

    if prediction[0] == 1:
        st.success("Student Likely to Purchase Course")
    else:
        st.error("Student Unlikely to Purchase Course")

    st.write(f"### Purchase Probability: {purchase_prob:.2f}%")

    st.progress(int(purchase_prob))


if os.path.exists("model.pkl"):
    model = pickle.load(open("model.pkl", "rb"))
else:
    st.error("Model file not found. Train the model first.")