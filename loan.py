
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import base64

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="wide")

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("loan_data.csv")
    df.drop(["Text"], axis=1, inplace=True)

    for col in ["Employment_Status", "Approval"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


    approved = df[df["Approval"] == 0]
    rejected = df[df["Approval"] == 1]

    rejected_sample = rejected.sample(len(approved), random_state=42)
    balanced_df = pd.concat([approved, rejected_sample])


    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    x = balanced_df.iloc[:, :-1]
    y = balanced_df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred) * 100
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    scores = cross_val_score(model, x, y, cv=5, scoring="accuracy")
    mean_cv_score = scores.mean()

    return model, x.columns, acc, conf_matrix, class_report, mean_cv_score, df

model, model_features, accuracy, conf_matrix, class_report, mean_cv_score,df = load_and_train_model()

print(df)



def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.jpg")



st.title("Loan Approval Prediction 🏦")
st.write(f"Model Accuracy: **{accuracy:.2f}%**")
st.write(f"Cross-Validation Score (Mean): **{mean_cv_score:.2f}**")

st.sidebar.header("Enter Applicant Details")

income = st.sidebar.number_input("Annual Income($)", min_value=0, step=1000)
credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, step=1)
loan_amount = st.sidebar.number_input("Loan Amount($)", min_value=0, step=1000)
dti_ratio = st.sidebar.slider("DTI Ratio (%)", min_value=0.0, max_value=100.0, step=0.1)
employment_status = st.sidebar.selectbox("Employment Status", options=["Employed", "Unemployed"])

employment_status_map = {"Employed": 0, "Unemployed": 1}
employment_status_numeric = employment_status_map[employment_status]

sample = pd.DataFrame(
    [[income, credit_score, loan_amount, dti_ratio, employment_status_numeric]],
    columns=model_features
)

if st.sidebar.button("Predict Loan Approval"):
    prediction = model.predict(sample)[0]
    if prediction == 0:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")

st.header("Model Performance Metrics")
st.subheader("Confusion Matrix")
st.write(conf_matrix)

st.subheader("Classification Report")
st.text(class_report)



