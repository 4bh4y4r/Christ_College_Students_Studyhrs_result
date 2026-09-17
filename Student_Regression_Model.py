
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    "StudyHours": [
        1, 2, 2, 3, 3,
        4, 4, 5, 5, 6,
        6, 7, 7, 8, 8,
        9, 9, 10, 10, 11
    ],

    "Attendance": [
        55, 60, 65, 60, 70,
        65, 75, 70, 80, 75,
        85, 80, 90, 85, 92,
        90, 95, 92, 96, 98
    ],

    "Result": [
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

X = df[["StudyHours", "Attendance"]]
y = df["Result"]

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

st.title("Student Pass/Fail Prediction")

hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

attendance = st.number_input(
    "Enter Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

if st.button("Predict"):

    student = [[hours, attendance]]

    prediction = model.predict(student)

    probability = model.predict_proba(student)

    if prediction[0] == 1:
        st.success("Student will PASS")
    else:
        st.error("Student will FAIL")

    st.write(
        "Probability of Pass:",
        round(probability[0][1] * 100, 2),
        "%"
    )
