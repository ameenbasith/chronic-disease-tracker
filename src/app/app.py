from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the predictor class
try:
    from models.predict_model import DiseaseProgressionPredictor
except ImportError:
    st.error("Could not import DiseaseProgressionPredictor. Using a placeholder instead.")


    # Define a simple placeholder class if import fails
    class DiseaseProgressionPredictor:
        def __init__(self, models_dir='../../models/'):
            pass

# Set page config
st.set_page_config(
    page_title="Chronic Disease Progression Tracker",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load data and models
@st.cache_data
def load_data():
    """Load the processed CKD dataset"""
    try:
        # Adjust the path based on your directory structure
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, "data", "processed", "ckd_features.csv")

        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.warning(f"Error loading data: {str(e)}")
        st.info("Creating a sample dataset for demonstration")

        # Create a sample dataset with necessary columns
        np.random.seed(42)
        n_samples = 100

        # Generate random data
        sample_data = {
            'age': np.random.normal(60, 15, n_samples).clip(18, 90).astype(int),
            'bp': np.random.normal(130, 20, n_samples).clip(90, 200).astype(int),
            'sc': np.random.normal(1.5, 1.0, n_samples).clip(0.5, 7.0),
            'hemo': np.random.normal(12, 2, n_samples).clip(6, 18),
            'pcv': np.random.normal(38, 6, n_samples).clip(20, 55).astype(int),
            'wc': np.random.normal(8000, 2000, n_samples).clip(3000, 15000),
            'rc': np.random.normal(4.5, 1.0, n_samples).clip(2.5, 7.0),
            'bu': np.random.normal(50, 25, n_samples).clip(10, 180).astype(int),
            'sod': np.random.normal(135, 5, n_samples).clip(120, 150).astype(int),
            'pot': np.random.normal(4.5, 1.0, n_samples).clip(2.5, 7.0),
            'bgr': np.random.normal(140, 60, n_samples).clip(70, 400).astype(int),
        }

        # Add categorical columns
        categorical_cols = {
            'class': ['ckd', 'notckd'],
            'htn': ['yes', 'no'],
            'dm': ['yes', 'no'],
            'cad': ['yes', 'no'],
            'pe': ['yes', 'no'],
            'ane': ['yes', 'no'],
            'rbc': ['normal', 'abnormal'],
            'pc': ['normal', 'abnormal'],
            'pcc': ['present', 'notpresent'],
            'ba': ['present', 'notpresent'],
            'appet': ['good', 'poor'],
        }

        # Generate random categorical data with appropriate distributions
        for col, values in categorical_cols.items():
            # Set probabilities to be reasonable for the column
            if col == 'class':
                # 60% ckd, 40% notckd
                probabilities = [0.6, 0.4]
            elif col in ['htn', 'dm']:
                # 40% yes, 60% no
                probabilities = [0.4, 0.6]
            elif col in ['cad', 'pe', 'ane']:
                # 25% yes, 75% no
                probabilities = [0.25, 0.75]
            else:
                # Equal probabilities for other categories
                probabilities = [1 / len(values)] * len(values)

            sample_data[col] = np.random.choice(values, n_samples, p=probabilities)

        # Add numerical columns with integers
        sample_data['al'] = np.random.choice(range(6), n_samples)
        sample_data['su'] = np.random.choice(range(6), n_samples)
        sample_data['sg'] = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_samples)

        # Create DataFrame
        df = pd.DataFrame(sample_data)

        # Calculate GFR based on age and serum creatinine
        df['gfr'] = 175 * (df['sc'] ** -1.154) * (df['age'] ** -0.203)

        # Create CKD stages based on GFR
        conditions = [
            df['gfr'] >= 90,
            (df['gfr'] >= 60) & (df['gfr'] < 90),
            (df['gfr'] >= 45) & (df['gfr'] < 60),
            (df['gfr'] >= 30) & (df['gfr'] < 45),
            (df['gfr'] >= 15) & (df['gfr'] < 30),
            df['gfr'] < 15
        ]
        choices = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
        df['ckd_stage'] = np.select(conditions, choices, default=np.nan)

        # Calculate risk score based on risk factors
        df['risk_score'] = 0
        risk_factors = ['htn', 'dm', 'cad', 'ane', 'pe']
        for factor in risk_factors:
            df.loc[df[factor] == 'yes', 'risk_score'] += 1

        # Create progression risk category
        df['progression_risk'] = 'low'
        df.loc[(df['risk_score'] >= 2) & (df['risk_score'] < 4), 'progression_risk'] = 'medium'
        df.loc[df['risk_score'] >= 4, 'progression_risk'] = 'high'

        # Save the sample data to the expected location
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            save_dir = os.path.join(base_dir, "data", "processed")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, "ckd_features.csv")
            df.to_csv(save_path, index=False)
            st.success("Created and saved sample dataset")
        except Exception as save_error:
            st.warning(f"Could not save sample dataset: {str(save_error)}")

        return df

@st.cache_resource
def load_models():
    """Load the trained models"""
    try:
        models_dir = "../../models/"
        predictor = DiseaseProgressionPredictor(models_dir=models_dir)
        return predictor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Main function
def main():
    """Main function to run the Streamlit app"""

    # Sidebar
    st.sidebar.title("Chronic Disease Progression Tracker")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Patient Dashboard", "Disease Progression", "Risk Assessment", "Intervention Recommendations"]
    )

    # Load data
    df = load_data()

    # Load models
    # Comment out for now as we haven't trained models yet
    predictor = load_models()

    # Display page based on selection
    if page == "Home":
        show_home_page(df)
    elif page == "Patient Dashboard":
        show_patient_dashboard(df)
    elif page == "Disease Progression":
        show_disease_progression(df)
    elif page == "Risk Assessment":
        show_risk_assessment(df)
    elif page == "Intervention Recommendations":
        show_intervention_recommendations(df)


def show_home_page(df):
    """Show the home page with overview statistics"""
    st.title("Chronic Kidney Disease Progression Tracker")
    st.markdown("""
    This application helps healthcare providers track and predict the progression of Chronic Kidney Disease (CKD) in patients.
    It provides visualizations, risk assessments, and personalized intervention recommendations based on patient data.

    ### Key Features
    - **Patient Dashboard**: View and manage individual patient data
    - **Disease Progression**: Visualize and predict disease progression paths
    - **Risk Assessment**: Identify high-risk patients and factors
    - **Intervention Recommendations**: Get personalized recommendations based on risk profiles

    ### Dataset Overview
    This application uses the Chronic Kidney Disease dataset from the UCI Machine Learning Repository, which contains data from 400 patients.
    """)

    # Display basic statistics
    st.subheader("Dataset Statistics")

    # Create 3 columns for basic stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Patients", f"{len(df)}")

    with col2:
        ckd_count = df[df['class'] == 'ckd'].shape[0]
        st.metric("CKD Patients", f"{ckd_count} ({ckd_count / len(df) * 100:.1f}%)")

    with col3:
        non_ckd_count = df[df['class'] == 'notckd'].shape[0]
        st.metric("Non-CKD Patients", f"{non_ckd_count} ({non_ckd_count / len(df) * 100:.1f}%)")

    # Display distribution of CKD stages
    st.subheader("CKD Stage Distribution")

    # Plot CKD stage distribution
    fig = px.histogram(
        df,
        x="ckd_stage",
        color="class",
        barmode="group",
        title="Distribution of CKD Stages",
        labels={"ckd_stage": "CKD Stage", "count": "Number of Patients", "class": "Diagnosis"},
        category_orders={"class": ["ckd", "notckd"]}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display risk factor distribution
    st.subheader("Risk Factor Distribution")

    # Create risk factor bar chart
    risk_cols = ['htn', 'dm', 'cad', 'ane', 'pe']
    risk_data = {}

    for col in risk_cols:
        if col in df.columns:
            risk_data[col] = df[df[col] == 'yes'].shape[0]

    risk_df = pd.DataFrame({
        'Risk Factor': [
            'Hypertension',
            'Diabetes Mellitus',
            'Coronary Artery Disease',
            'Anemia',
            'Pedal Edema'
        ],
        'Count': list(risk_data.values())
    })

    fig = px.bar(
        risk_df,
        x="Risk Factor",
        y="Count",
        title="Prevalence of Risk Factors",
        color="Count",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_patient_dashboard(df):
    """Show the patient dashboard with individual patient data"""
    st.title("Patient Dashboard")

    # Patient selector
    st.sidebar.subheader("Select Patient")

    # Create a patient ID selector
    patient_ids = list(range(1, len(df) + 1))
    selected_patient_id = st.sidebar.selectbox("Patient ID", patient_ids)

    # Get selected patient data (index is 0-based, so subtract 1)
    patient_data = df.iloc[selected_patient_id - 1]

    # Display patient information
    st.subheader(f"Patient ID: {selected_patient_id}")

    # Create 3 columns for basic info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Age", f"{patient_data['age']:.0f} years")
        st.metric("Blood Pressure", f"{patient_data['bp']:.0f} mmHg")
        st.metric("CKD Status", "CKD" if patient_data['class'] == 'ckd' else "Not CKD")

    with col2:
        st.metric("GFR", f"{patient_data['gfr']:.1f} mL/min")
        st.metric("CKD Stage", f"Stage {patient_data['ckd_stage']}")
        st.metric("Hemoglobin", f"{patient_data['hemo']:.1f} g/dL")

    with col3:
        st.metric("Risk Score", f"{patient_data['risk_score']:.0f}/5")
        st.metric("Progression Risk", patient_data['progression_risk'].title())
        st.metric("Diabetes", "Yes" if patient_data['dm'] == 'yes' else "No")

    # Display clinical parameters
    st.subheader("Clinical Parameters")

    # Create tabs for different parameter groups
    tab1, tab2, tab3 = st.tabs(["Laboratory Values", "Clinical Signs", "Risk Factors"])

    with tab1:
        # Create 3 columns for lab values
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Serum Creatinine", f"{patient_data['sc']:.2f} mg/dL")
            st.metric("Blood Urea", f"{patient_data['bu']:.0f} mg/dL")
            st.metric("Sodium", f"{patient_data['sod']:.0f} mEq/L")

        with col2:
            st.metric("Potassium", f"{patient_data['pot']:.1f} mEq/L")
            st.metric("Hemoglobin", f"{patient_data['hemo']:.1f} g/dL")
            st.metric("Packed Cell Volume", f"{patient_data['pcv']:.0f}%")

        with col3:
            st.metric("White Blood Cells", f"{patient_data['wc']:.0f} cells/mm³")
            st.metric("Red Blood Cells", f"{patient_data['rc']:.1f} million/mm³")
            st.metric("Blood Glucose", f"{patient_data['bgr']:.0f} mg/dL")

    with tab2:
        # Create 2 columns for clinical signs
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Specific Gravity", f"{patient_data['sg']}")
            st.metric("Albumin", f"{patient_data['al']}")
            st.metric("Sugar", f"{patient_data['su']}")

        with col2:
            st.metric("Red Blood Cells", patient_data['rbc'].title())
            st.metric("Pus Cells", patient_data['pc'].title())
            st.metric("Appetite", patient_data['appet'].title())

    with tab3:
        # Create a comparison of risk factors
        risk_factors = {
            'Hypertension': patient_data['htn'] == 'yes',
            'Diabetes': patient_data['dm'] == 'yes',
            'Coronary Artery Disease': patient_data['cad'] == 'yes',
            'Anemia': patient_data['ane'] == 'yes',
            'Pedal Edema': patient_data['pe'] == 'yes'
        }

        # Plot risk factors as a horizontal bar chart
        risk_df = pd.DataFrame({
            'Risk Factor': list(risk_factors.keys()),
            'Present': list(risk_factors.values())
        })

        fig = px.bar(
            risk_df,
            y="Risk Factor",
            x=[1] * len(risk_factors),
            color="Present",
            orientation='h',
            title="Risk Factor Profile",
            labels={"x": "Status", "Present": "Present"},
            color_discrete_map={True: "red", False: "green"}
        )

        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Absent', 'Present'],
                range=[0, 1]
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    # Display simulated progression data (this would ideally come from time series data)
    st.subheader("Simulated Disease Progression")

    # Create simulated GFR data based on patient's risk profile
    dates = [datetime.now() - timedelta(days=30*i) for i in range(12, -1, -1)]

    # Base GFR on patient's current GFR and add some random variation with a downward trend
    base_gfr = patient_data['gfr']

    # Calculate decline rate based on risk score (higher risk = faster decline)
    decline_rate = 0.3 * (patient_data['risk_score'] + 1)

    # Generate simulated GFR values with some random noise
    np.random.seed(selected_patient_id)  # Use patient ID as seed for reproducibility
    gfr_values = [max(15, base_gfr - (decline_rate * i) + np.random.normal(0, 2)) for i in range(13)]

    # Create a DataFrame for the time series
    gfr_df = pd.DataFrame({
        'Date': dates,
        'GFR': gfr_values
    })

    # Plot GFR time series
    fig = px.line(
        gfr_df,
        x="Date",
        y="GFR",
        title="GFR Progression Over Time",
        labels={"GFR": "GFR (mL/min)", "Date": "Date"}
    )

    # Add horizontal lines for CKD stage thresholds
    fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Stage 1", annotation_position="left")
    fig.add_hline(y=60, line_dash="dash", line_color="yellow", annotation_text="Stage 2", annotation_position="left")
    fig.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="Stage 3a", annotation_position="left")
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Stage 3b", annotation_position="left")
    fig.add_hline(y=15, line_dash="dash", line_color="darkred", annotation_text="Stage 4", annotation_position="left")

    st.plotly_chart(fig, use_container_width=True)


def show_disease_progression(df):
    """Show disease progression analysis and predictions"""
    st.title("Disease Progression Analysis")

    st.markdown("""
    This page provides insights into disease progression patterns and predictions based on patient characteristics.
    """)

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Population Overview", "Progression Factors", "Survival Analysis"])

    with tab1:
        st.subheader("Population Overview")

        # Distribution of GFR by CKD stage
        fig = px.box(
            df,
            x="ckd_stage",
            y="gfr",
            color="class",
            title="GFR Distribution by CKD Stage",
            labels={"ckd_stage": "CKD Stage", "gfr": "GFR (mL/min)", "class": "Diagnosis"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Create a scatter plot of GFR vs. age
        fig = px.scatter(
            df,
            x="age",
            y="gfr",
            color="class",
            size="risk_score",
            hover_data=["ckd_stage", "htn", "dm"],
            title="GFR vs. Age by Diagnosis",
            labels={"age": "Age (years)", "gfr": "GFR (mL/min)", "class": "Diagnosis"}
        )

        # Add reference lines for CKD stages
        fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Stage 1")
        fig.add_hline(y=60, line_dash="dash", line_color="yellow", annotation_text="Stage 2")
        fig.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="Stage 3a")
        fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Stage 3b")
        fig.add_hline(y=15, line_dash="dash", line_color="darkred", annotation_text="Stage 4")

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Progression Risk Factors")

        # Create a correlation heatmap for numerical variables
        numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'hemo', 'pcv', 'gfr', 'risk_score', 'ckd_stage']
        corr_df = df[numeric_cols].corr()

        fig = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            title="Correlation Between Clinical Parameters",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)

        # Create a bar chart for risk factors by progression risk
        risk_factors = ['htn', 'dm', 'cad', 'ane', 'pe']

        # Prepare data for chart
        risk_data = []

        for factor in risk_factors:
            for risk in ['low', 'medium', 'high']:
                subset = df[df['progression_risk'] == risk]
                if subset.shape[0] > 0:  # Check if subset is not empty
                    count = subset[subset[factor] == 'yes'].shape[0]
                    percentage = count / subset.shape[0] * 100
                    risk_data.append({
                        'Risk Factor': {'htn': 'Hypertension', 'dm': 'Diabetes',
                                        'cad': 'CAD', 'ane': 'Anemia', 'pe': 'Edema'}[factor],
                        'Progression Risk': risk.title(),
                        'Percentage': percentage
                    })

        risk_chart_df = pd.DataFrame(risk_data)

        fig = px.bar(
            risk_chart_df,
            x="Risk Factor",
            y="Percentage",
            color="Progression Risk",
            barmode="group",
            title="Prevalence of Risk Factors by Progression Risk",
            labels={"Percentage": "Percentage of Patients (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Survival Analysis")

        st.markdown("""
        This section provides a simulation of CKD progression over time based on risk factors.
        In a real application, this would be based on actual longitudinal data and survival analysis models.
        """)

        # Create simulated survival curves based on risk categories
        time_points = list(range(0, 61, 6))  # 0 to 60 months in 6-month intervals

        # Create survival probabilities based on risk categories
        # These are simulated values for illustration
        survival_high = [1, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.40, 0.35, 0.30, 0.25]
        survival_medium = [1, 0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]
        survival_low = [1, 0.99, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82]

        # Create a DataFrame for plotting
        survival_df = pd.DataFrame({
            'Time (months)': time_points * 3,
            'Progression-Free Probability': survival_high + survival_medium + survival_low,
            'Risk Category': ['High'] * len(time_points) + ['Medium'] * len(time_points) + ['Low'] * len(time_points)
        })

        # Plot survival curves
        fig = px.line(
            survival_df,
            x="Time (months)",
            y="Progression-Free Probability",
            color="Risk Category",
            title="Simulated CKD Progression-Free Survival by Risk Category",
            labels={"Progression-Free Probability": "Probability of Not Progressing to Next Stage"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add explanation
        st.markdown("""
        The chart above shows the estimated probability of a patient not progressing to the next CKD stage over time, 
        based on their risk category. Patients with high risk factors have a steeper decline in kidney function over time.

        **Note:** These are simulated probabilities for illustration. In a real application, these would be calculated 
        using Cox Proportional Hazards models or other survival analysis techniques on longitudinal patient data.
        """)


def show_risk_assessment(df):
    """Show risk assessment for CKD progression"""
    st.title("Risk Assessment")

    st.markdown("""
    This page provides tools to assess the risk of CKD progression based on patient characteristics.
    """)

    # Risk calculator
    st.subheader("CKD Progression Risk Calculator")

    # Create two columns for the form
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        bp = st.number_input("Blood Pressure (mmHg)", min_value=60, max_value=250, value=120)
        sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.2, step=0.1)
        htn = st.checkbox("Hypertension")
        dm = st.checkbox("Diabetes Mellitus")

    with col2:
        hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=12.0, step=0.1)
        pcv = st.number_input("Packed Cell Volume (%)", min_value=10, max_value=60, value=40)
        pe = st.checkbox("Pedal Edema")
        ane = st.checkbox("Anemia")
        cad = st.checkbox("Coronary Artery Disease")

    # Calculate GFR and risk score when button is clicked
    if st.button("Calculate Risk"):
        # Calculate GFR using the simplified MDRD formula
        gfr = 175 * (sc ** -1.154) * (age ** -0.203)

        # Determine CKD stage
        if gfr >= 90:
            ckd_stage = 1
        elif gfr >= 60:
            ckd_stage = 2
        elif gfr >= 45:
            ckd_stage = 3
        elif gfr >= 30:
            ckd_stage = 3.5
        elif gfr >= 15:
            ckd_stage = 4
        else:
            ckd_stage = 5

        # Calculate risk score
        risk_score = sum([htn, dm, cad, ane, pe])

        # Determine progression risk
        if risk_score >= 4:
            progression_risk = "High"
        elif risk_score >= 2:
            progression_risk = "Medium"
        else:
            progression_risk = "Low"

        # Display results
        st.subheader("Risk Assessment Results")

        # Create 3 columns for results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Estimated GFR", f"{gfr:.1f} mL/min")
            st.metric("CKD Stage", f"Stage {ckd_stage}")

        with col2:
            st.metric("Risk Score", f"{risk_score}/5")
            st.metric("Progression Risk", progression_risk)

        with col3:
            # Categorize risk of progression to next stage
            if progression_risk == "High":
                next_stage_risk = "High"
                time_estimate = "6-12 months"
            elif progression_risk == "Medium":
                next_stage_risk = "Moderate"
                time_estimate = "1-3 years"
            else:
                next_stage_risk = "Low"
                time_estimate = "3+ years"

            st.metric("Risk of Progression", next_stage_risk)
            st.metric("Estimated Time to Next Stage", time_estimate)

        # Show gauge chart for risk visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 2], 'color': 'green'},
                    {'range': [2, 4], 'color': 'yellow'},
                    {'range': [4, 5], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display recommendations based on risk
        st.subheader("Recommendations")

        if progression_risk == "High":
            st.markdown("""
            ### High Risk of Progression

            **Recommended Monitoring:**
            - Monitor GFR, creatinine, and electrolytes every 1-3 months
            - Protein/creatinine ratio every 3 months
            - BP checks every 2 weeks

            **Interventions to Consider:**
            - Optimize blood pressure control (target <130/80 mmHg)
            - ACE inhibitor or ARB therapy if not contraindicated
            - Glycemic control for diabetic patients (target HbA1c <7%)
            - Dietary protein restriction (0.8 g/kg/day)
            - Nephrology referral if not already under specialist care
            """)

        elif progression_risk == "Medium":
            st.markdown("""
            ### Medium Risk of Progression

            **Recommended Monitoring:**
            - Monitor GFR, creatinine, and electrolytes every 3-6 months
            - Protein/creatinine ratio every 6 months
            - BP checks monthly

            **Interventions to Consider:**
            - Blood pressure control (target <140/90 mmHg)
            - ACE inhibitor or ARB therapy if albuminuria present
            - Dietary sodium restriction
            - Lifestyle modifications (exercise, weight management)
            - Consider nephrology referral
            """)

        else:
            st.markdown("""
            ### Low Risk of Progression

            **Recommended Monitoring:**
            - Monitor GFR, creatinine, and electrolytes every 6-12 months
            - Annual protein/creatinine ratio
            - Regular BP checks

            **Interventions to Consider:**
            - Maintain blood pressure within normal range
            - Healthy lifestyle (diet, exercise, avoid nephrotoxins)
            - Manage cardiovascular risk factors
            - Annual follow-up with primary care
            """)

    # Population risk distribution
    st.subheader("Population Risk Distribution")

    # Create a scatter plot of risk score vs. CKD stage
    fig = px.scatter(
        df,
        x="ckd_stage",
        y="risk_score",
        color="progression_risk",
        size="gfr",
        hover_data=["age", "htn", "dm"],
        title="Risk Score vs. CKD Stage",
        labels={
            "ckd_stage": "CKD Stage",
            "risk_score": "Risk Score",
            "progression_risk": "Progression Risk",
            "gfr": "GFR"
        },
        color_discrete_map={
            "high": "red",
            "medium": "yellow",
            "low": "green"
        }
    )
    st.plotly_chart(fig, use_container_width=True)


def show_intervention_recommendations(df):
    """Show intervention recommendations based on risk profiles"""
    st.title("Intervention Recommendations")

    st.markdown("""
    This page provides personalized intervention recommendations based on patient risk profiles.
    """)

    # Patient selector for recommendations
    st.sidebar.subheader("Select Patient")

    # Create a patient ID selector
    patient_ids = list(range(1, len(df) + 1))
    selected_patient_id = st.sidebar.selectbox("Patient ID", patient_ids, key="rec_patient_id")

    # Get selected patient data (index is 0-based, so subtract 1)
    patient_data = df.iloc[selected_patient_id - 1]

    # Display patient information
    st.subheader(f"Patient ID: {selected_patient_id}")

    # Create 3 columns for basic info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Age", f"{patient_data['age']:.0f} years")
        st.metric("GFR", f"{patient_data['gfr']:.1f} mL/min")
        st.metric("CKD Stage", f"Stage {patient_data['ckd_stage']}")

    with col2:
        st.metric("Risk Score", f"{patient_data['risk_score']:.0f}/5")
        st.metric("Progression Risk", patient_data['progression_risk'].title())
        st.metric("CKD Status", "CKD" if patient_data['class'] == 'ckd' else "Not CKD")

    with col3:
        st.metric("Hypertension", "Yes" if patient_data['htn'] == 'yes' else "No")
        st.metric("Diabetes", "Yes" if patient_data['dm'] == 'yes' else "No")
        st.metric("Anemia", "Yes" if patient_data['ane'] == 'yes' else "No")

    # Generate personalized recommendations
    st.subheader("Personalized Recommendations")

    # Create tabs for different recommendation categories
    tab1, tab2, tab3, tab4 = st.tabs(["Treatment", "Monitoring", "Lifestyle", "Education"])

    with tab1:
        st.markdown("### Treatment Recommendations")

        # Recommendations based on risk factors and CKD stage
        recommendations = []

        # CKD stage-based recommendations
        if patient_data['ckd_stage'] >= 4:
            recommendations.append(
                "**Nephrology Referral**: Urgent nephrology referral for specialized care and preparation for renal replacement therapy.")
        elif patient_data['ckd_stage'] >= 3:
            recommendations.append("**Nephrology Referral**: Consider nephrology referral for co-management.")

        # Blood pressure recommendations
        if patient_data['htn'] == 'yes':
            recommendations.append(
                "**Blood Pressure Management**: Target BP <130/80 mmHg with ACE inhibitor or ARB as first-line therapy.")
        else:
            recommendations.append("**Blood Pressure Management**: Maintain BP <140/90 mmHg with regular monitoring.")

        # Diabetes recommendations
        if patient_data['dm'] == 'yes':
            recommendations.append(
                "**Diabetes Management**: Target HbA1c <7% with appropriate glycemic control to minimize kidney damage.")

        # Anemia recommendations
        if patient_data['ane'] == 'yes':
            recommendations.append(
                "**Anemia Management**: Evaluate iron status and consider erythropoiesis-stimulating agents if hemoglobin <10 g/dL.")

        # Medication recommendations based on risk
        if patient_data['progression_risk'] == 'high':
            recommendations.append(
                "**SGLT2 Inhibitors**: Consider for their renoprotective effects in patients with diabetes or high cardiovascular risk.")
            recommendations.append(
                "**Mineralocorticoid Receptor Antagonists**: Consider for additional proteinuria reduction in resistant cases.")

        # Display recommendations
        for rec in recommendations:
            st.markdown(rec)

    with tab2:
        st.markdown("### Monitoring Recommendations")

        # Monitoring frequency based on CKD stage and risk
        if patient_data['ckd_stage'] >= 4:
            monitoring_freq = "Monthly"
        elif patient_data['ckd_stage'] >= 3 or patient_data['progression_risk'] == 'high':
            monitoring_freq = "Every 3 months"
        else:
            monitoring_freq = "Every 6 months"

        st.markdown(f"**Monitoring Frequency**: {monitoring_freq}")

        # Create a monitoring schedule table
        monitoring_data = {
            "Parameter": [
                "Serum Creatinine/GFR",
                "Electrolytes (Na, K)",
                "Urinalysis",
                "Urine Protein/Creatinine Ratio",
                "Blood Pressure",
                "Complete Blood Count",
                "Nutritional Assessment",
                "Cardiovascular Risk Assessment"
            ],
            "Frequency": [
                monitoring_freq,
                monitoring_freq,
                monitoring_freq,
                monitoring_freq,
                "Every visit/Home monitoring",
                "Every 3-6 months",
                "Every 6 months",
                "Annually"
            ],
            "Notes": [
                "Trend is important; rapid decline needs urgent evaluation",
                "Watch for hyperkalemia, especially with RAAS blockers",
                "Monitor for hematuria and pyuria",
                "Target <0.5 g/g for optimal outcomes",
                f"Target <{'130' if patient_data['ckd_stage'] >= 2 else '140'}/80 mmHg",
                "Monitor for anemia development/progression",
                "Protein, calorie, and salt intake assessment",
                "Including lipids, glucose, and other CVD risk factors"
            ]
        }

        monitoring_df = pd.DataFrame(monitoring_data)
        st.table(monitoring_df)

    with tab3:
        st.markdown("### Lifestyle Recommendations")

        # Dietary recommendations
        st.markdown("#### Dietary Recommendations")

        if patient_data['ckd_stage'] >= 4:
            protein_rec = "Restrict protein intake to 0.6-0.8 g/kg/day under dietitian supervision"
        else:
            protein_rec = "Moderate protein intake to 0.8 g/kg/day"

        salt_rec = "Restrict sodium to <2g/day (about 5g of salt)"

        if patient_data['ckd_stage'] >= 3:
            potassium_rec = "Monitor and potentially restrict potassium intake based on serum levels"
            phosphate_rec = "Restrict phosphate intake and consider phosphate binders if levels elevated"
        else:
            potassium_rec = "No specific restrictions unless hyperkalemia develops"
            phosphate_rec = "No specific restrictions unless hyperphosphatemia develops"

        diet_recs = [
            f"**Protein**: {protein_rec}",
            f"**Sodium**: {salt_rec}",
            f"**Potassium**: {potassium_rec}",
            f"**Phosphate**: {phosphate_rec}",
            "**Fluid Intake**: Maintain adequate hydration unless fluid restriction advised",
            "**General Diet**: Emphasize fruits, vegetables, whole grains, and lean proteins; limit processed foods"
        ]

        for rec in diet_recs:
            st.markdown(rec)

        # Exercise recommendations
        st.markdown("#### Exercise Recommendations")

        if patient_data['ckd_stage'] >= 4:
            exercise_rec = "Light to moderate exercise as tolerated; consider physical therapy consultation"
        else:
            exercise_rec = "Regular moderate exercise (150 minutes/week) including aerobic and resistance training"

        st.markdown(f"**Exercise Regimen**: {exercise_rec}")
        st.markdown("**Benefits**: Improves cardiovascular health, insulin sensitivity, and blood pressure control")

        # Other lifestyle recommendations
        st.markdown("#### Additional Lifestyle Modifications")

        additional_recs = [
            "**Smoking Cessation**: Critical for slowing CKD progression and reducing cardiovascular risk",
            "**Alcohol**: Limit to moderate consumption or avoid completely",
            "**Weight Management**: Maintain healthy BMI (18.5-24.9 kg/m²)",
            "**Sleep**: Aim for 7-8 hours of quality sleep; screen for sleep apnea which is common in CKD"
        ]

        for rec in additional_recs:
            st.markdown(rec)

    with tab4:
        st.markdown("### Patient Education Resources")

        education_resources = [
            {
                "Title": "Understanding Your Kidney Function",
                "Description": "Basic information about CKD and GFR interpretation",
                "Format": "Printable PDF, Video"
            },
            {
                "Title": "Medication Management in CKD",
                "Description": "Guide to medication adherence and avoiding nephrotoxic drugs",
                "Format": "Printable PDF, Mobile App"
            },
            {
                "Title": "Kidney-Friendly Diet",
                "Description": "Dietary guidelines specific to CKD stage with meal plans",
                "Format": "Printable PDF, Cookbook, Mobile App"
            },
            {
                "Title": "Monitoring Your Blood Pressure",
                "Description": "Home BP monitoring instructions and log sheets",
                "Format": "Printable PDF, Mobile App"
            },
            {
                "Title": "Living Well with Kidney Disease",
                "Description": "Strategies for managing lifestyle with CKD",
                "Format": "Support Group, Online Forum"
            }
        ]

        # Convert to DataFrame and display as table
        education_df = pd.DataFrame(education_resources)
        st.table(education_df)

        # Display links to kidney disease organizations
        st.markdown("### Kidney Disease Organizations")

        org_links = [
            "[National Kidney Foundation](https://www.kidney.org)",
            "[American Association of Kidney Patients](https://aakp.org)",
            "[National Institute of Diabetes and Digestive and Kidney Diseases](https://www.niddk.nih.gov/health-information/kidney-disease)",
            "[Kidney Disease: Improving Global Outcomes (KDIGO)](https://kdigo.org)"
        ]

        for link in org_links:
            st.markdown(link)

    # Display predicted progression trajectory
    st.subheader("Predicted Disease Progression Trajectory")

    # Create a projection of GFR decline based on risk factors
    # This would ideally come from a trained model

    # Base prediction on current GFR, age, and risk score
    current_gfr = patient_data['gfr']
    risk_score = patient_data['risk_score']
    age = patient_data['age']

    # Calculate annual GFR decline rate based on risk score
    # Higher risk = faster decline
    # These are simplified estimates for demonstration
    if risk_score >= 4:  # High risk
        annual_decline = 4 + (risk_score - 3)  # 5-6 mL/min/year
    elif risk_score >= 2:  # Medium risk
        annual_decline = 2 + (risk_score - 1)  # 3-4 mL/min/year
    else:  # Low risk
        annual_decline = 1 + risk_score  # 1-2 mL/min/year

    # Adjust for age (older patients tend to have slower absolute decline)
    if age > 65:
        annual_decline = annual_decline * 0.8

    # Create time points for the next 10 years
    years = list(range(11))  # 0 to 10 years

    # Calculate projected GFR values
    gfr_projection = [max(0, current_gfr - (annual_decline * year)) for year in years]

    # Calculate projected CKD stages
    stage_projection = []
    for gfr in gfr_projection:
        if gfr >= 90:
            stage_projection.append(1)
        elif gfr >= 60:
            stage_projection.append(2)
        elif gfr >= 45:
            stage_projection.append(3)
        elif gfr >= 30:
            stage_projection.append(3.5)
        elif gfr >= 15:
            stage_projection.append(4)
        else:
            stage_projection.append(5)

    # Create a DataFrame for the projection
    projection_df = pd.DataFrame({
        'Year': years,
        'Projected GFR': gfr_projection,
        'Projected CKD Stage': stage_projection
    })

    # Plot GFR projection
    fig = px.line(
        projection_df,
        x="Year",
        y="Projected GFR",
        title="Projected GFR Decline Over the Next 10 Years",
        labels={"Projected GFR": "GFR (mL/min)", "Year": "Years from Now"}
    )

    # Add horizontal lines for CKD stage thresholds
    fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Stage 1", annotation_position="left")
    fig.add_hline(y=60, line_dash="dash", line_color="yellow", annotation_text="Stage 2", annotation_position="left")
    fig.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="Stage 3a", annotation_position="left")
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Stage 3b", annotation_position="left")
    fig.add_hline(y=15, line_dash="dash", line_color="darkred", annotation_text="Stage 4", annotation_position="left")

    st.plotly_chart(fig, use_container_width=True)

    # Show estimated time to key events
    est_time_to_next_stage = None
    est_time_to_kidney_failure = None

    # Find time to next stage
    current_stage = stage_projection[0]
    for i, stage in enumerate(stage_projection):
        if stage > current_stage:
            est_time_to_next_stage = i
            break

    # Find time to kidney failure (Stage 5)
    for i, stage in enumerate(stage_projection):
        if stage == 5:
            est_time_to_kidney_failure = i
            break

    # Create columns for key events
    col1, col2 = st.columns(2)

    with col1:
        if est_time_to_next_stage:
            st.metric("Estimated Time to Next CKD Stage", f"{est_time_to_next_stage} years")
        else:
            st.metric("Estimated Time to Next CKD Stage", ">10 years")

    with col2:
        if est_time_to_kidney_failure:
            st.metric("Estimated Time to Kidney Failure", f"{est_time_to_kidney_failure} years")
        else:
            st.metric("Estimated Time to Kidney Failure", ">10 years")

    # Disclaimer
    st.markdown("""
    **Disclaimer**: These projections are estimates based on population data and the patient's current risk factors. 
    Individual progression can vary significantly. Regular monitoring and adjusting treatment based on actual progression 
    is essential for optimal care.
    """)


# Run the main app
if __name__ == "__main__":
    main()