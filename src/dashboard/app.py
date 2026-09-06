"""
Streamlit dashboard for PharmaPulse — Patient Journey Intelligence & Next-Best-Action Engine.
Provides interactive visualization and interaction with the PharmaPulse system.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our modules
try:
    from src.models.next_best_action import NextBestActionEngine
    from src.models.evidence_retrieval.evidence_retriever import ClinicalGuidelinesEvidenceRetriever
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    MODELS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="PharmaPulse Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    .guideline-box {
        background-color: #f0fff8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load models and initialize engine."""
    if not MODELS_AVAILABLE:
        return None, None

    try:
        engine = NextBestActionEngine()
        evidence_retriever = ClinicalGuidelinesEvidenceRetriever()
        return engine, evidence_retriever
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def main():
    """Main dashboard application."""
    # Header
    st.markdown('<h1 class="main-header">💊 PharmaPulse Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Patient Journey Intelligence & Next-Best-Action Engine</p>', unsafe_allow_html=True)

    # Load models
    engine, evidence_retriever = load_models()

    if engine is None:
        st.warning("⚠️ Models not loaded. Some functionality may be limited.")
        # Show demo mode
        show_demo_mode()
        return

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Patient Overview", "Next-Best-Action Recommendations", "Evidence Retrieval", "Population Analytics", "Model Performance"]
    )

    # Page routing
    if page == "Patient Overview":
        show_patient_overview(engine)
    elif page == "Next-Best-Action Recommendations":
        show_recommendations(engine)
    elif page == "Evidence Retrieval":
        show_evidence_retrieval(evidence_retriever)
    elif page == "Population Analytics":
        show_population_analytics(engine)
    elif page == "Model Performance":
        show_model_performance(engine)


def show_demo_mode():
    """Show demo mode when models aren't available."""
    st.info("📝 Running in demo mode. To enable full functionality, ensure all models are trained and available.")

    # Show sample data
    st.subheader("Sample Patient Data")
    sample_data = pd.DataFrame({
        'patient_id': [f'P{i:05d}' for i in range(10)],
        'age': np.random.randint(18, 80, 10),
        'gender': np.random.choice(['Male', 'Female'], 10),
        'current_treatment': np.random.choice([1, 2, 0], 10),
        'adherence_risk': np.random.rand(10),
        'discontinuation_risk': np.random.rand(10) * 0.3
    })

    st.dataframe(sample_data)

    # Sample visualizations
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(sample_data, x='age', nbins=10, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(sample_data, x='adherence_risk', y='discontinuation_risk',
                        color='gender', title="Risk Scores")
        st.plotly_chart(fig, use_container_width=True)


def show_patient_overview(engine):
    """Show patient overview page."""
    st.header("👥 Patient Overview")

    # Patient selector
    patient_ids = engine.patient_data['patient_id'].tolist()
    selected_patient = st.selectbox("Select Patient", patient_ids)

    if selected_patient:
        # Get patient data
        patient_data = engine.patient_data[engine.patient_data['patient_id'] == selected_patient].iloc[0]

        # Display patient info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Patient ID", selected_patient)
            st.metric("Age", f"{patient_data['age']} years")
            st.metric("Gender", patient_data['gender'])

        with col2:
            st.metric("Ethnicity", patient_data['ethnicity'])
            st.metric("Current Treatment", f"Treatment {patient_data['treatment']}")
            st.metric("Cluster", f"Cluster {patient_data['cluster']}" if not pd.isna(patient_data['cluster']) else "Not assigned")

        with col3:
            # Get risks
            adherence_risks = engine.predict_adherence_risk(selected_patient)
            discontinuation_risks = engine.predict_discontinuation_risk(selected_patient)

            avg_adherence_risk = np.mean([v['adherence_risk'] for v in adherence_risks.values()]) if adherence_risks else 0.5
            avg_discontinuation_risk = np.mean([v['discontinuation_risk'] for v in discontinuation_risks.values()]) if discontinuation_risks else 0.5

            st.metric("Adherence Risk", f"{avg_adherence_risk:.2f}")
            st.metric("Discontinuation Risk", f"{avg_discontinuation_risk:.2f}")

            # Risk level indicators
            adherence_level = "High" if avg_adherence_risk > 0.5 else "Moderate" if avg_adherence_risk > 0.3 else "Low"
            discontinuation_level = "High" if avg_discontinuation_risk > 0.5 else "Moderate" if avg_discontinuation_risk > 0.2 else "Low"

            st.markdown(f"**Adherence Level:** {adherence_level}")
            st.markdown(f"**Discontinuation Level:** {discontinuation_level}")

        # Risk visualization
        if adherence_risks or discontinuation_risks:
            st.subheader("Risk Predictions by Model")

            # Prepare data for plotting
            risk_data = []
            for model_name, risk_dict in adherence_risks.items():
                risk_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Risk Type': 'Adherence Risk',
                    'Risk Score': risk_dict['adherence_risk']
                })

            for model_name, risk_dict in discontinuation_risks.items():
                risk_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Risk Type': 'Discontinuation Risk',
                    'Risk Score': risk_dict['discontinuation_risk']
                })

            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                fig = px.bar(risk_df, x='Model', y='Risk Score', color='Risk Type',
                            barmode='group', title="Risk Predictions by Model")
                st.plotly_chart(fig, use_container_width=True)


def show_recommendations(engine):
    """Show next-best-action recommendations page."""
    st.header("🎯 Next-Best-Action Recommendations")

    # Patient selector
    patient_ids = engine.patient_data['patient_id'].tolist()
    selected_patient = st.selectbox("Select Patient for Recommendation", patient_ids)

    if selected_patient:
        # Generate recommendation
        if st.button("Generate Recommendation", type="primary"):
            with st.spinner("Generating recommendation..."):
                recommendation = engine.recommend_next_best_action(selected_patient)

                # Display recommendation
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.subheader(f"Recommendation for {selected_patient}")
                st.write(f"**Action:** {recommendation['recommended_action']}")
                st.write(f"**Details:** {recommendation['action_details']}")
                st.markdown('</div>', unsafe_allow_html=True)

                # Display risk scores
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Adherence Risk", f"{recommendation['risk_scores']['adherence_risk']:.3f}")
                with col2:
                    st.metric("Discontinuation Risk", f"{recommendation['risk_scores']['discontinuation_risk']:.3f}")

                # Display supporting evidence
                st.subheader("Supporting Evidence")

                # Risk scores from individual models
                if recommendation['supporting_evidence']['adherence_risks']:
                    st.write("**Adherence Risk Models:**")
                    for model_name, risk_data in recommendation['supporting_evidence']['adherence_risks'].items():
                        st.write(f"- {model_name}: {risk_data['adherence_risk']:.3f}")

                if recommendation['supporting_evidence']['discontinuation_risks']:
                    st.write("**Discontinuation Risk Models:**")
                    for model_name, risk_data in recommendation['supporting_evidence']['discontinuation_risks'].items():
                        st.write(f"- {model_name}: {risk_data['discontinuation_risk']:.3f}")

                # Uplift estimates
                if recommendation['supporting_evidence']['uplift_treatment_a'] or recommendation['supporting_evidence']['uplift_treatment_b']:
                    st.write("**Treatment Uplift Estimates:**")
                    if recommendation['supporting_evidence']['uplift_treatment_a']:
                        avg_uplift_a = np.mean([v.get('uplift', 0) for v in recommendation['supporting_evidence']['uplift_treatment_a'].values()])
                        st.write(f"- Treatment A Uplift: {avg_uplift_a:.3f}")
                    if recommendation['supporting_evidence']['uplift_treatment_b']:
                        avg_uplift_b = np.mean([v.get('uplift', 0) for v in recommendation['supporting_evidence']['uplift_treatment_b'].values()])
                        st.write(f"- Treatment B Uplift: {avg_uplift_b:.3f}")

                # Clinical guidelines
                guidelines = recommendation['supporting_evidence'].get('clinical_guidelines', [])
                if guidelines:
                    st.write("**Retrieved Clinical Guidelines:**")
                    for i, guideline in enumerate(guidelines, 1):
                        st.markdown(f'''
                        <div class="guideline-box">
                            <strong>{i}. {guideline["title"]}</strong> (Similarity: {guideline["similarity_score"]:.3f})<br>
                            {guideline["content"][:200]}...
                        </div>
                        ''', unsafe_allow_html=True)


def show_evidence_retrieval(evidence_retriever):
    """Show evidence retrieval page."""
    st.header("📚 Clinical Guidelines Evidence Retrieval")

    # Query input
    st.subheader("Search Clinical Guidelines")
    query = st.text_input("Enter your query (e.g., 'patient with hypertension and high adherence risk')",
                         placeholder="e.g., elderly patient with diabetes concerns")
    top_k = st.slider("Number of results to return", 1, 5, 3)

    if st.button("Search Guidelines", type="primary") and query:
        with st.spinner("Searching guidelines..."):
            results = evidence_retriever.retrieve_evidence(query, top_k=top_k)

            if results:
                st.subheader(f"Search Results for: '{query}'")
                for i, result in enumerate(results, 1):
                    st.markdown(f'''
                    <div class="guideline-box">
                        <strong>{i}. {result["title"]}</strong> (Similarity: {result["similarity_score"]:.3f})<br>
                        <em>Condition:</em> {result.get("condition", "N/A")} |
                        <em>Category:</em> {result.get("category", "N/A")} |
                        <em>Source:</em> {result.get("source", "N/A")}<br>
                        {result["content"]}
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No guidelines found matching your query.")

    # Patient-based retrieval
    st.subheader("Retrieve Guidelines Based on Patient Data")
    st.write("Enter patient information to retrieve relevant clinical guidelines:")

    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=50)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        patient_conditions = st.text_input("Conditions (comma-separated)", placeholder="e.g., hypertension, diabetes")

    with col2:
        adherence_risk = st.slider("Adherence Risk", 0.0, 1.0, 0.5, 0.01)
        discontinuation_risk = st.slider("Discontinuation Risk", 0.0, 1.0, 0.1, 0.01)
        current_treatment = st.selectbox("Current Treatment", [0, 1, 2], format_func=lambda x: f"Treatment {x}" if x > 0 else "No Treatment")

    if st.button("Get Patient-Specific Guidelines", type="primary"):
        # Prepare patient data
        conditions_list = [c.strip() for c in patient_conditions.split(",")] if patient_conditions else []
        patient_data = {
            "age": patient_age,
            "gender": patient_gender,
            "conditions": conditions_list,
            "adherence_risk": adherence_risk,
            "discontinuation_risk": discontinuation_risk,
            "current_treatment": current_treatment
        }

        with st.spinner("Retrieving patient-specific guidelines..."):
            results = evidence_retriever.retrieve_evidence_for_patient(patient_data, top_k=3)

            if results:
                st.subheader("Patient-Specific Clinical Guidelines")
                for i, result in enumerate(results, 1):
                    st.markdown(f'''
                    <div class="guideline-box">
                        <strong>{i}. {result["title"]}</strong> (Similarity: {result["similarity_score"]:.3f})<br>
                        <em>Condition:</em> {result.get("condition", "N/A")} |
                        <em>Category:</em> {result.get("category", "N/A")} |
                        <em>Source:</em> {result.get("source", "N/A")}<br>
                        {result["content"]}
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No guidelines found for the specified patient profile.")


def show_population_analytics(engine):
    """Show population analytics page."""
    st.header("📊 Population Analytics")

    # Population statistics
    st.subheader("Population Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(engine.patient_data))
    with col2:
        st.metric("Female Patients", len(engine.patient_data[engine.patient_data['gender'] == 'Female']))
    with col3:
        st.metric("Male Patients", len(engine.patient_data[engine.patient_data['gender'] == 'Male']))
    with col4:
        st.metric("Avg Age", f"{engine.patient_data['age'].mean():.1f} years")

    # Distribution plots
    st.subheader("Demographic Distributions")

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution
        fig = px.histogram(engine.patient_data, x='age', nbins=20, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Gender distribution
        gender_counts = engine.patient_data['gender'].value_counts()
        fig = px.pie(values=gender_counts.values, names=gender_counts.index, title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Treatment distribution
    if 'treatment' in engine.patient_data.columns:
        st.subheader("Treatment Distribution")
        treatment_counts = engine.patient_data['treatment'].value_counts()
        treatment_labels = [f"Treatment {x}" if x > 0 else "No Treatment" for x in treatment_counts.index]
        fig = px.bar(x=treatment_labels, y=treatment_counts.values, title="Treatment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Cluster distribution
    if 'cluster' in engine.patient_data.columns and not engine.patient_data['cluster'].isna().all():
        st.subheader("Patient Segmentation Clusters")
        cluster_counts = engine.patient_data['cluster'].value_counts().sort_index()
        fig = px.bar(x=[f"Cluster {x}" for x in cluster_counts.index], y=cluster_counts.values, title="Patient Cluster Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Risk distribution (if available)
    st.subheader("Risk Score Distribution Sample")
    # Show risk distribution for a sample of patients
    sample_size = min(100, len(engine.patient_data))
    sample_patients = engine.patient_data['patient_id'].head(sample_size)

    adherence_risks_sample = []
    discontinuation_risks_sample = []

    for patient_id in sample_patients:
        adherence_risks = engine.predict_adherence_risk(patient_id)
        discontinuation_risks = engine.predict_discontinuation_risk(patient_id)

        avg_adherence = np.mean([v['adherence_risk'] for v in adherence_risks.values()]) if adherence_risks else 0.5
        avg_discontinuation = np.mean([v['discontinuation_risk'] for v in discontinuation_risks.values()]) if discontinuation_risks else 0.5

        adherence_risks_sample.append(avg_adherence)
        discontinuation_risks_sample.append(avg_discontinuation)

    risk_df = pd.DataFrame({
        'Patient_ID': sample_patients,
        'Adherence_Risk': adherence_risks_sample,
        'Discontinuation_Risk': discontinuation_risks_sample
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(risk_df, x='Adherence_Risk', nbins=20, title="Adherence Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(risk_df, x='Discontinuation_Risk', nbins=20, title="Discontinuation Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Risk correlation
    fig = px.scatter(risk_df, x='Adherence_Risk', y='Discontinuation_Risk',
                    title="Adherence vs Discontinuation Risk Correlation",
                    trendline="ols")
    st.plotly_chart(fig, use_container_width=True)


def show_model_performance(engine):
    """Show model performance page."""
    st.header("📈 Model Performance")

    st.subheader("Available Models")

    # Display model information
    if 'predictive' in engine.models:
        st.write("**Predictive Models:**")
        for model_name in engine.models['predictive'].keys():
            st.write(f"- {model_name.replace('_', ' ').title()}")

    if 'survival' in engine.models:
        st.write("**Survival Models:**")
        for model_name in engine.models['survival'].keys():
            st.write(f"- {model_name.replace('_', ' ').title()}")

    if 'uplift' in engine.models:
        st.write("**Uplift Models:**")
        for model_name in engine.models['uplift'].keys():
            st.write(f"- {model_name.replace('_', ' ').title()}")

    # Model training info (if available)
    st.subheader("Model Information")
    st.info("""
    **Model Details:**
    - Predictive Models: Logistic Regression, XGBoost, LightGBM for adherence and discontinuation risk
    - Survival Models: Kaplan-Meier estimator, Cox Proportional Hazards model
    - Uplift Models: Two-model approach for treatment effect estimation
    - Feature Engineering: Longitudinal patient features including demographics, treatment exposure, adherence metrics, clinical trends
    - Patient Segmentation: KMeans clustering with PCA visualization
    """)

    # Feature importance (if available from tree-based models)
    st.subheader("Feature Importance (Sample)")
    st.info("Feature importance visualization would be displayed here for tree-based models (XGBoost, LightGBM)")

    # SHAP values placeholder
    st.subheader("Model Explainability")
    st.info("""
    SHAP (SHapley Additive exPlanations) values would be displayed here to show:
    - Feature contributions to individual predictions
    - Global feature importance
    - Dependence plots and interaction effects
    """)

    # Model performance metrics
    st.subheader("Performance Metrics")
    st.info("""
    In a production setting, this section would display:
    - AUC-ROC, accuracy, precision, recall, F1-score for classification models
    - Concordance index for survival models
    - Calibration plots and calibration curves
    - Cross-validation results
    """)


if __name__ == "__main__":
    main()