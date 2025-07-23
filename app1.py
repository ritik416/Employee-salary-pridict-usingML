import streamlit as st
import pandas as pd
import joblib

# 🚀 Load trained model & encoders
model = joblib.load("best_model.pkl")
le_education = joblib.load("education_encoder.pkl")
le_occupation = joblib.load("occupation_encoder.pkl")
le_gender = joblib.load("gender_encoder.pkl")
# 🔥 CUSTOM CSS 🔥
st.markdown(
    """
    <style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    h1 {
        color: #FF4B4B;
    }

    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: None;
        border-radius: 8px;
        padding: 0.5em 1em;
    }

    .stSidebar {
        background-color: #262730;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# 🚀 Streamlit page config
st.set_page_config(
    page_title="Employee Income Classification",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Income Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# 🚀 Sidebar Inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox(
    "Education Level",
    list(le_education.classes_)
)
occupation = st.sidebar.selectbox(
    "Occupation",
    list(le_occupation.classes_)
)
gender = st.sidebar.selectbox(
    "Gender",
    list(le_gender.classes_)
)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)

# 🚀 Encode inputs
education_encoded = le_education.transform([education])[0]
occupation_encoded = le_occupation.transform([occupation])[0]
gender_encoded = le_gender.transform([gender])[0]

# 🚀 Make input DataFrame for prediction
input_df = pd.DataFrame({
    'age': [age],
    'education': [education_encoded],
    'occupation': [occupation_encoded],
    'gender': [gender_encoded],
    'hours-per-week': [hours_per_week]
})

st.write("### Input Data")
st.write(input_df)

# 🚀 Single Prediction
if st.button("Predict Income Class"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success("✅ Prediction: Income > 50K")
    else:
        st.success("✅ Prediction: Income ≤ 50K")

# 🚀 Batch Prediction
st.markdown("---")
st.markdown("### 📄 Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", batch_data.head())

    # Encode batch columns
    try:
        batch_data['education'] = le_education.transform(batch_data['education'])
        batch_data['occupation'] = le_occupation.transform(batch_data['occupation'])
        batch_data['gender'] = le_gender.transform(batch_data['gender'])
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    X_batch = batch_data[['age', 'education', 'occupation', 'gender', 'hours-per-week']]
    batch_preds = model.predict(X_batch)
    batch_data['PredictedClass'] = batch_preds
    batch_data['PredictedClass'] = batch_data['PredictedClass'].map({1: '>50K', 0: '<=50K'})

    st.write("Predictions:")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Predictions CSV",
        csv,
        file_name='predicted_classes.csv',
        mime='text/csv'
    )
