# Course Purchase Prediction
This project builds a **Machine Learning model** to predict whether a student is likely to purchase an online course based on their learning behavior and activity on an EdTech platform.

The trained model is deployed using **Streamlit**, allowing users to input student details and receive **real-time predictions along with purchase probability**.

---

## Project Objective

EdTech platforms want to identify students who are most likely to purchase courses.  
By analyzing student engagement and learning behavior, this system predicts the **likelihood of course purchase**.

This helps platforms improve:

- Targeted marketing strategies
- Course recommendation systems
- Student engagement analysis

---

## Dataset Features

The dataset contains student activity information:

| Feature | Description |
|------|------|
| age | Age of the student |
| study_hours_per_week | Weekly study time |
| previous_courses_completed | Number of courses previously completed |
| platform_visits_per_month | Number of times the platform was visited |
| assignment_completion_rate | Percentage of assignments completed |
| purchased_course | Target variable (0 = No, 1 = Yes) |

---

## Machine Learning Model

A **Random Forest Classifier** is used to train the prediction model.
## Advantages of Using Random Forest

Random Forest was chosen because it provides several benefits:

✔ High prediction accuracy  
✔ Resistant to overfitting  
✔ Works well with tabular datasets  
✔ Handles nonlinear relationships between variables  
✔ Provides probability predictions  
✔ Requires minimal preprocessing  

These properties make Random Forest an excellent choice for predicting **student purchase behavior based on engagement data**.

---

## Application in This Project

In this system, the Random Forest model analyzes relationships between student engagement metrics such as:

- study hours
- platform activity
- assignment completion
- past course history

Using these features, the model learns patterns that indicate **which students are more likely to purchase a course**.

The trained model is then integrated into a **Streamlit web application** where users can input student information and receive real-time predictions.

Steps performed:

1. Data loading and exploration
2. Data preprocessing
3. Train-test split
4. Model training
5. Model evaluation using accuracy
6. Model serialization using `pickle`

---

## Streamlit Web Application

The project includes an interactive **Streamlit web interface** where users can:

✔ Enter student activity details  
✔ Predict course purchase likelihood  
✔ View purchase probability using a visual progress meter  

## How to Run the Project

### Install Dependencies

```bash
pip install streamlit pandas scikit-learn numpy
python train_model.py
streamlit run app.py
