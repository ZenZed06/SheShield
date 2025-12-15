import joblib
import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF

# Load model, scaler, and features
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="SheShield", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
<style>
/* Page background */
.stApp {
    background: linear-gradient(180deg, #F5C7D9, #fff5fa);
}

/* Titles */
.title {
    text-align: center;
    font-size: 64px;
    font-weight: 800;
    color: #d63384;
    font-family: 'Poppins', 'Segoe UI', sans-serif;
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    color: #BA044D;
    margin-top: 10px;
    font-family: 'Poppins', 'Segoe UI', sans-serif;
    letter-spacing: 0.5px;
}
.info-box {
    background: linear-gradient(135deg, #FFC0CB, #FF69B4);
    padding: 25px;
    border-radius: 20px;
    color: #4A0028;
    font-family: 'Poppins', sans-serif;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.2);
    text-align: center;
    line-height: 1.6;
    margin-bottom: 30px;
}
.info-box h3 { font-size: 24px; font-weight: 800; margin-bottom: 10px; }
.info-box p { font-size: 18px; font-weight: 500; }

.start-title {
    font-family: 'Poppins', sans-serif;
    font-size: 40px;
    font-weight: 800;
    color: #d63384;
    text-align: center;
    margin-top: 40px;
    margin-bottom: 20px;
    letter-spacing: 1px;
}
.start-subtitle {
    font-family: 'Poppins', sans-serif;
    font-size: 22px;
    font-weight: 500;
    color: #9A6E84;
    text-align: center;
    margin-bottom: 40px;
    line-height: 1.5;
}
.custom-title {
    color: #FF69B4;  
    font-family: 'Arial', sans-serif;  
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- Titles ----------
st.markdown('<div class="title">SheShield: Breast Health Assessment Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="margin-bottom:20px;">Designed for healthcare professionals — provides insights, not a standalone diagnosis.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>Protect, Respect, and Empower Women </h3>
    <p>Early awareness of breast health can save lives. This tool assists healthcare professionals in identifying potential risks, providing actionable insights to guide timely interventions and personalized care. For professional use only.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="start-title">Breast Health Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="start-subtitle">Upload Patient Data below to generate risk insights for clinical evaluation.</div>', unsafe_allow_html=True)






# ---------- ADD GUIDE HERE ----------
# Sample CSV download
sample_data = pd.DataFrame([["example"] * len(feature_names)], columns=feature_names)
sample_csv = sample_data.to_csv(index=False)
st.download_button(
    label="Download Sample CSV",
    data=sample_csv,
    file_name="sample_patient_data.csv",
    mime="text/csv"
)

# Required features info
st.info(
    f"📄 **CSV Requirements:**\n\n"
    f"- Upload data for **one patient at a time**.\n"
    f"- Include all required columns: **{', '.join(feature_names)}**\n"
    f"- Ensure all feature values are numeric.\n\n"
    f"💡 You can download a sample CSV to see the proper format."
)

# ---------- Patient Name ----------
st.markdown('<div class="custom-title">Enter Patient Name</div>', unsafe_allow_html=True)
patient_name = st.text_input("")

if patient_name and not patient_name.replace(" ", "").isalpha():
    st.error("Patient name can only contain alphabets and spaces.")
    patient_name = ""

# ---------- File Upload ----------
st.markdown('<div class="custom-title">Upload Patient Data (CSV or Excel file)</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv", "xlsx"])
input_df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        # Check missing columns
        missing_cols = [col for col in feature_names if col not in input_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            input_df = None
        else:
            input_df = input_df[feature_names]

            # Validate single patient input
            if len(input_df) != 1:
                st.error("Please upload data for **only one patient at a time**.")
                input_df = None
            else:
                st.success("File uploaded successfully!")
                st.dataframe(input_df.style.set_properties(**{'color': '#FFFFFF'}))

    except Exception as e:
        st.error(f"Error reading the file: {e}")

# ---------- Generate Risk Insights ----------
if st.button("Generate Risk Insights"):
    if input_df is None:
        st.error("Please upload a CSV or Excel file before generating risk insights.")
    elif not patient_name:
        st.error("Please enter a valid patient name (alphabets only).")
    else:
        try:
            # Ensure numeric data
            model_input = input_df[feature_names].apply(pd.to_numeric, errors='coerce')
            if model_input.isnull().any().any():
                st.error("All input features must be numeric.")
            else:
                with st.spinner("Generating risk insights..."):
                    input_scaled = scaler.transform(model_input)

                    # Predictions
                    predictions = model.predict(input_scaled)
                    probabilities = model.predict_proba(input_scaled)[:, 1]

                    # Risk & Prediction
                    risk = round(probabilities[0] * 100, 1)
                    prediction = "High Risk (Malignant)" if predictions[0] == 1 else "Low Risk (Benign)"

                    # After generating risk & prediction
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.set_text_color(255, 0, 100)

                    pdf.cell(0, 10, "SheShield: Patient Risk Assessment Report", ln=True, align='C')
                    pdf.ln(10)

                    pdf.set_font("Arial", '', 12)
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
                    pdf.cell(0, 10, f"Risk of Malignancy: {risk}%", ln=True)
                    pdf.cell(0, 10, f"Assessment: {prediction}", ln=True)
                    pdf.ln(5)
                    pdf.multi_cell(0, 10, "This report is intended for healthcare professionals for clinical use only.")
                    pdf.output("report.pdf")  # Temporary file

                    # Provide download in Streamlit
                    with open("report.pdf", "rb") as f:
                        st.download_button(
                            label="Download Risk Report (PDF)",
                            data=f,
                            file_name=f"{patient_name}_risk_report.pdf",
                            mime="application/pdf"
                        )

                    # Report Header
                    st.markdown("<h3 style='color:#FF69B4;'>Patient Risk Assessment Report</h3>", unsafe_allow_html=True)

                    # Individual Report
                    if prediction.startswith("High Risk"):
                        st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #FFC0CB, #FF69B4);
                                padding: 25px;
                                border-radius: 20px;
                                color: #4A0028;
                                font-family: 'Poppins', sans-serif;
                                box-shadow: 0px 5px 15px rgba(0,0,0,0.2);
                                text-align: left;
                                margin-bottom: 20px;">
                                <p><strong>⚠️ This model is 98.25% accurate</strong></p>
                                <h3>Patient: {patient_name}</h3>
                                <p><strong>Risk of Malignancy:</strong> {risk}%</p>
                                <p><strong>Assessment:</strong> Recommended: consult a qualified physician immediately for diagnostic evaluation, further testing, and care planning.</p>
                                <p>This report is intended for healthcare professionals for clinical use only.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #90EE90, #32CD32);
                                padding: 25px;
                                border-radius: 20px;
                                color: #004d00;
                                font-family: 'Poppins', sans-serif;
                                box-shadow: 0px 5px 15px rgba(0,0,0,0.2);
                                text-align: left;
                                margin-bottom: 20px;">
                                <p><strong>✅ This model is 98.25% accurate</strong></p>
                                <h3>Patient: {patient_name}</h3>
                                <p><strong>Risk of Malignancy:</strong> {risk}%</p>
                                <p><strong>Assessment:</strong> Good news! Continue routine monitoring and maintain breast health awareness.</p>
                                <p>This report is intended for healthcare professionals for clinical use only.</p>
                            </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating predictions: {e}")

