import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.markdown("""
<style>

.main {
background-color:#0E1117;
}

.block-container{
padding-top:2rem;
}

.metric-card{
background: rgba(255,255,255,0.05);
padding:20px;
border-radius:15px;
box-shadow:0px 0px 10px rgba(0,0,0,0.3);
text-align:center;
}

.section-title{
font-size:28px;
font-weight:bold;
margin-top:20px;
margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="EdTech Purchase Prediction",
    page_icon= "🎓",
    layout="wide"
)


model_data = pickle.load(open("model.pkl","rb"))
model = model_data["model"]
accuracy = model_data["accuracy"]


st.title("EdTech Student Purchase Prediction Dashboard")
st.markdown(
"""
Predict the likelihood of a student purchasing a course based on learning behavior,
platform engagement, and academic activity.
"""
)


st.sidebar.header("Student Activity")
age = st.sidebar.slider("Age",18,60,22)
study_hours = st.sidebar.slider(
"Study Hours per Week",0,40,12
)
courses_completed = st.sidebar.slider(
"Previous Courses Completed",0,10,2
)
platform_visits = st.sidebar.slider(
"Platform Visits per Month",0,50,15
)
assignment_rate = st.sidebar.slider(
"Assignment Completion Rate",0,100,80
)
predict = st.sidebar.button("Predict Purchase")


col1,col2,col3,col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card"><h3>Model</h3><p>Random Forest</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>Dataset</h3><p>500 Students</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>Features</h3><p>5 Inputs</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(
        f'<div class="metric-card"><h3>Accuracy</h3><p>{accuracy * 100:.2f}%</p></div>',unsafe_allow_html=True)
st.divider()


engagement_score = (
study_hours*0.4 +
platform_visits*0.3 +
assignment_rate*0.3
)
st.subheader("Student Engagement Score")
st.progress(int(engagement_score))
st.write(f"Engagement Score: **{engagement_score:.2f}**")


if predict:
    features = np.array([[age,
                          study_hours,
                          courses_completed,
                          platform_visits,
                          assignment_rate]])

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]*100
    left,right = st.columns([2,1])

    with left:
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"High likelihood of purchase ({probability:.2f}%)")
        else:
            st.warning(f"Low likelihood of purchase ({probability:.2f}%)")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text':"Purchase Probability %"},
            gauge={'axis':{'range':[0,100]}}
        ))
        st.plotly_chart(fig,use_container_width=True)

    with right:
        st.subheader("Student Summary")
        st.write(f"Age: **{age}**")
        st.write(f"Study Hours: **{study_hours}**")
        st.write(f"Courses Completed: **{courses_completed}**")
        st.write(f"Platform Visits: **{platform_visits}**")
        st.write(f"Assignment Completion: **{assignment_rate}%**")

engagement_df = pd.DataFrame({
"Metric":["Study Hours","Platform Visits","Assignments"],
"Value":[study_hours,platform_visits,assignment_rate]
})

fig = px.pie(
engagement_df,
values="Value",
names="Metric",
hole=0.6,
title="Student Engagement Distribution"
)
st.plotly_chart(fig,use_container_width=True)



st.divider()
st.subheader("Student Activity Analysis")
activity_data = pd.DataFrame({
"Feature":[
"Study Hours",
"Courses Completed",
"Platform Visits",
"Assignment Completion"
],

"Value":[
study_hours,
courses_completed,
platform_visits,
assignment_rate
]
})

fig = px.bar(
activity_data,
x="Feature",
y="Value",
color="Feature",
title="Student Learning Behavior"
)
st.plotly_chart(fig,use_container_width=True)

st.divider()
st.subheader("Feature Importance")
importance = model.feature_importances_

features = [
"Age",
"Study Hours",
"Courses Completed",
"Platform Visits",
"Assignment Completion"
]

importance_df = pd.DataFrame({
"Feature":features,
"Importance":importance
})

fig2 = px.bar(
importance_df,
x="Importance",
y="Feature",
orientation="h",
color="Importance",
title="Which Factors Influence Purchase Most"
)

st.plotly_chart(fig2,use_container_width=True)
st.divider()
st.caption("Machine Learning Model deployed using Streamlit | EdTech Analytics Dashboard")
