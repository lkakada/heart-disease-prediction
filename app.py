# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------- Streamlit Page Config ---------------- #
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Predictor")
st.write("This app predicts the risk of heart disease based on patient data.")

# ---------------- Load CSV ---------------- #
DATA_PATH = "heart.csv"
try:
    df = pd.read_csv(DATA_PATH)

    # Ensure numeric integer target for palette matching
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)

    required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                     'restecg', 'thalach', 'exang', 'oldpeak',
                     'slope', 'ca', 'thal', 'target']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain: {required_cols}")
        st.stop()

    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    # Dataset description
    st.caption("""
    **Dataset Preview**  
    This table shows the first few rows of the **UCI Heart Disease dataset**, which is used to train and evaluate the prediction model. 
    Each row represents a patient record, and each column contains a medical measurement or diagnostic indicator relevant to heart disease risk assessment.

    - **age** – Age of the patient in years  
    - **sex** – 1 = Male, 0 = Female  
    - **cp** – Chest pain type (0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic)  
    - **trestbps** – Resting blood pressure in mm Hg  
    - **chol** – Serum cholesterol in mg/dL  
    - **fbs** – Fasting blood sugar > 120 mg/dL (1 = Yes, 0 = No)  
    - **restecg** – Resting electrocardiogram results (0–2)  
    - **thalach** – Maximum heart rate achieved during exercise  
    - **exang** – Exercise-induced angina (1 = Yes, 0 = No)  
    - **oldpeak** – ST depression induced by exercise relative to rest  
    - **slope** – Slope of peak exercise ST segment (0–2)  
    - **ca** – Number of major vessels colored by fluoroscopy (0–3)  
    - **thal** – Thalassemia status (1–3)  
    - **target** – Prediction label (1 = Heart disease, 0 = No disease)  

    This preview confirms that the correct dataset has been loaded and allows verification of data quality before running predictions or generating visualizations.
    """)


    # Sidebar Inputs
    st.sidebar.header("Input Patient Data")

    def user_input():
        age = st.sidebar.slider("Age", int(df.age.min()), int(df.age.max()), int(df.age.mean()))
        sex = st.sidebar.radio("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.sidebar.selectbox("Chest Pain Type (cp)", sorted(df.cp.unique()))
        trestbps = st.sidebar.slider("Resting Blood Pressure", int(df.trestbps.min()), int(df.trestbps.max()), int(df.trestbps.mean()))
        chol = st.sidebar.slider("Cholesterol (mg/dL)", int(df.chol.min()), int(df.chol.max()), int(df.chol.mean()))
        fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.sidebar.selectbox("Resting ECG", sorted(df.restecg.unique()))
        thalach = st.sidebar.slider("Max Heart Rate", int(df.thalach.min()), int(df.thalach.max()), int(df.thalach.mean()))
        exang = st.sidebar.radio("Exercise-Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.sidebar.slider("ST Depression", float(df.oldpeak.min()), float(df.oldpeak.max()), float(df.oldpeak.mean()), 0.1)
        slope = st.sidebar.selectbox("Slope of ST Segment", sorted(df.slope.unique()))
        ca = st.sidebar.selectbox("Number of Major Vessels", sorted(df.ca.unique()))
        thal = st.sidebar.selectbox("Thalassemia", sorted(df.thal.unique()))

        return pd.DataFrame({
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
            'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
            'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
        })

    input_df = user_input()

    # Model Training
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_scaled = scaler.transform(input_df)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0][1]

    # Prediction Output
    st.markdown("## 🔍 Prediction Result")

    if prediction[0] == 0:
        st.markdown(
            f"""
            <div style='background-color:#d4edda; padding:20px; border-radius:10px;'>
                <h2 style='color:green; text-align:center;'>🟢 No Heart Disease Detected</h2>
                <p style='font-size:18px; text-align:center;'>
                    <b>Probability of Heart Disease:</b> <span style='font-size:22px; color:green;'><b>{proba:.2%}</b></span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='background-color:#f8d7da; padding:20px; border-radius:10px;'>
                <h2 style='color:red; text-align:center;'>🔴 At Risk of Heart Disease</h2>
                <p style='font-size:18px; text-align:center;'>
                    <b>Probability of Heart Disease:</b> <span style='font-size:22px; color:red;'><b>{proba:.2%}</b></span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Shared Palette
    target_palette = {0: 'green', 1: 'red'}

    # Charts
    st.markdown("---")
    st.subheader("1. Correlation Heatmap")
    st.caption("Shows the strength of relationships between all features in the dataset. "
               "Darker colors indicate stronger positive or negative correlations with heart disease.")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
    st.pyplot(fig1)

    st.subheader("2. Cholesterol by Heart Disease (Box Plot)")
    st.caption("Compares cholesterol levels for patients with and without heart disease. "
               "Higher cholesterol values are often associated with increased heart disease risk.")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='target', y='chol', hue='target', palette=target_palette, legend=False, ax=ax2)
    ax2.set_xticks([0, 1])  # Explicit tick positions
    ax2.set_xticklabels(['No Disease (0)', 'Heart Disease (1)'])  # Matching labels
    st.pyplot(fig2)

    st.subheader("3. Age vs Max Heart Rate (Scatter Plot)")
    st.caption("Plots patient age against their maximum heart rate achieved. "
               "Lower maximum heart rates for older patients can be an indicator of potential heart issues.")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='age', y='thalach', hue='target', palette=target_palette, ax=ax3)
    st.pyplot(fig3)

    st.subheader("4. Resting Blood Pressure Distribution (Histogram)")
    st.caption("Shows how resting blood pressure values are distributed for patients with and without heart disease. "
               "Higher resting blood pressure can be a contributing risk factor.")
    fig4, ax4 = plt.subplots()
    sns.histplot(data=df, x='trestbps', hue='target', kde=True, multiple='stack', palette=target_palette, ax=ax4)
    st.pyplot(fig4)
    # ---------------- Additional Risk Factor Charts ---------------- #

    st.subheader("5. Chest Pain Type vs Heart Disease (Bar Chart)")
    st.caption(
        "Chest pain type (cp) is one of the strongest indicators of heart disease in this dataset. "
        "Certain types of chest pain, such as typical angina, are more strongly associated with positive diagnoses."
    )
    fig5, ax5 = plt.subplots()
    sns.countplot(data=df, x='cp', hue='target', palette=target_palette, ax=ax5)
    ax5.set_xlabel("Chest Pain Type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal, 3 = Asymptomatic)", labelpad=15)
    plt.tight_layout()
    ax5.set_ylabel("Number of Patients")
    ax5.legend(title="Heart Disease", labels=["No", "Yes"])
    st.pyplot(fig5)

    st.subheader("6. Fasting Blood Sugar vs Heart Disease (Bar Chart)")
    st.caption(
        "High fasting blood sugar (>120 mg/dL) can be a sign of diabetes or pre-diabetes, "
        "which increases cardiovascular risk. This chart compares heart disease prevalence "
        "between patients with normal and high fasting blood sugar."
    )
    fig6, ax6 = plt.subplots()
    sns.countplot(data=df, x='fbs', hue='target', palette=target_palette, ax=ax6)
    ax6.set_xlabel("Fasting Blood Sugar > 120 mg/dL (0 = No, 1 = Yes)")
    ax6.set_ylabel("Number of Patients")
    ax6.legend(title="Heart Disease", labels=["No", "Yes"])
    st.pyplot(fig6)

    # Footer
    st.markdown("---")
    st.markdown("### ℹ️ About This App")
    st.info("""
    This heart disease prediction app uses a Logistic Regression model trained on anonymized patient data 
    (from the UCI Heart Disease dataset). You can adjust parameters on the left to simulate different scenarios 
    and get a heart disease risk prediction.
    """)

except FileNotFoundError:
    st.error("❌ Could not find 'heart.csv'. Please place it in the same folder as this app.")
