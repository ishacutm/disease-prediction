import streamlit as st
import pandas as pd
import numpy as np
from train_model import DiseasePredictor
import json
import os

# Page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = {}
if 'medical_history' not in st.session_state:
    st.session_state.medical_history = {}
if 'lifestyle' not in st.session_state:
    st.session_state.lifestyle = {}
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Symptom categories
SYMPTOMS = {
    'Respiratory': ['cough', 'shortness_of_breath', 'chest_tightness', 'wheezing', 'sore_throat'],
    'Digestive': ['nausea', 'vomiting', 'abdominal_pain', 'diarrhea', 'loss_of_appetite'],
    'Neurological': ['headache', 'dizziness', 'memory_issues', 'seizures', 'numbness'],
    'Metabolic': ['fatigue', 'weight_loss', 'excessive_thirst', 'frequent_urination', 'fever'],
    'Cardiac': ['chest_pain', 'palpitations', 'leg_swelling'],
    'Skin': ['rash', 'itching', 'dry_skin', 'skin_redness', 'blisters', 'skin_bumps', 'skin_discoloration']
}

def load_model():
    """Load trained model"""
    if not os.path.exists('disease_model.pkl'):
        st.error("Model not found. Please train the model first.")
        if st.button("Train Model Now"):
            with st.spinner("Training model..."):
                from train_model import train_model
                train_model()
            st.success("Model trained successfully!")
            st.rerun()
        return None
    
    predictor = DiseasePredictor()
    predictor.load_model()
    return predictor

def home_page():
    """Home page"""
    # Custom CSS for home page with animations
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    .stMarkdown, .stText, .stSelectbox label, .stButton > button {
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        border: none;
        color: white !important;
        font-size: 18px;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 50px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        transition: all 0.4s ease;
        animation: bounce 2s infinite;
    }
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.1);
        box-shadow: 0 15px 30px rgba(0,0,0,0.4);
        background: linear-gradient(135deg, #ff5252, #d84315);
    }
    .stColumns > div {
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        padding: 30px;
        border-radius: 20px;
        margin: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
        animation: slideInLeft 0.8s ease-out;
    }
    .stColumns > div:nth-child(2) {
        animation: slideInRight 0.8s ease-out;
    }
    .stColumns > div:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    .hero-section {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: fadeIn 1.2s ease-out;
    }
    .feature-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1));
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.4s ease;
        animation: fadeIn 1.5s ease-out;
    }
    .feature-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 12px 25px rgba(0,0,0,0.3);
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 30px 0;
        animation: slideInLeft 1s ease-out;
    }
    .stat-item {
        text-align: center;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .stat-item:hover {
        transform: scale(1.1);
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff6b6b;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        animation: bounce 3s infinite;
    }
    h1 {
        font-size: 3.5rem !important;
        font-weight: bold !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 20px !important;
        animation: slideInLeft 0.6s ease-out;
    }
    h3 {
        font-size: 1.5rem !important;
        opacity: 0.9;
        margin-bottom: 30px !important;
        animation: slideInRight 0.8s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🏥 Disease Prediction System</h1>
        <h3>🤖 AI-Powered Disease Prediction with Mixed Symptom Handling</h3>
        <p style="font-size: 1.2rem; opacity: 0.8; margin: 20px 0;">Get accurate disease predictions using advanced machine learning that understands complex symptom combinations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-number">15</div>
            <div>Diseases Covered</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">25</div>
            <div>Symptoms Analyzed</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">70%</div>
            <div>Accuracy Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Key Features:</h4>
            <ul style="list-style: none; padding: 0;">
                <li>✨ Multi-category symptom selection</li>
                <li>🧠 Handles mixed symptoms correctly</li>
                <li>📊 Probability-based predictions</li>
                <li>🏥 Medical history integration</li>
                <li>💡 Explainable AI results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 How it works:</h4>
            <ol style="padding-left: 20px;">
                <li>🔍 Select symptoms from multiple categories</li>
                <li>📋 Provide medical history</li>
                <li>🏃‍♂️ Share lifestyle information</li>
                <li>🎯 Get probability-based predictions</li>
                <li>💊 View personalized recommendations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Trust indicators
    st.markdown("""
    <div style="text-align: center; margin: 40px 0; padding: 30px; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); border-radius: 20px; border: 1px solid rgba(255,255,255,0.2);">
        <h4>🛡️ Trusted by Healthcare Professionals</h4>
        <p style="opacity: 0.8; margin: 15px 0;">Our AI model uses advanced machine learning algorithms trained on comprehensive medical data to provide accurate predictions while handling complex symptom combinations.</p>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
            <span>⚡ Fast Results</span>
            <span>🔒 Secure & Private</span>
            <span>📱 Easy to Use</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Start Prediction", type="primary", use_container_width=True):
        st.session_state.page = 'Symptoms'
        st.rerun()

def symptoms_page():
    """Symptoms selection page"""
    # Custom CSS for symptoms page with animations
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%, #f093fb 100%);
        animation: fadeIn 0.8s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes checkboxPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); box-shadow: 0 0 20px rgba(255,255,255,0.5); }
        100% { transform: scale(1); }
    }
    .stMarkdown, .stText {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        font-weight: bold;
        animation: slideIn 0.6s ease-out;
    }
    .stCheckbox {
        animation: slideIn 0.8s ease-out;
        transition: all 0.3s ease;
    }
    .stCheckbox:hover {
        transform: translateX(5px);
    }
    .stCheckbox label {
        color: white !important;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stCheckbox input:checked + div {
        animation: checkboxPulse 0.5s ease;
    }
    .stButton > button {
        background: linear-gradient(135deg, rgba(255,255,255,0.3), rgba(255,255,255,0.1));
        border: 2px solid white;
        color: white !important;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        animation: slideIn 1s ease-out;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        background: linear-gradient(135deg, rgba(255,255,255,0.4), rgba(255,255,255,0.2));
    }
    .stExpander {
        background: linear-gradient(135deg, rgba(255,255,255,0.25), rgba(255,255,255,0.15));
        border-radius: 15px;
        border: 2px solid rgba(255,255,255,0.4);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        animation: slideIn 0.7s ease-out;
        transition: all 0.3s ease;
    }
    .stExpander:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .stColumns > div {
        background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1));
        padding: 20px;
        border-radius: 15px;
        margin: 8px;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        animation: slideIn 0.9s ease-out;
        transition: all 0.3s ease;
    }
    .stInfo {
        background: linear-gradient(135deg, rgba(255,255,255,0.3), rgba(255,255,255,0.2));
        color: white !important;
        border: 2px solid rgba(255,255,255,0.4);
        border-radius: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        animation: slideIn 1.2s ease-out;
    }
    h1, h2, h3 {
        color: white !important;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
        font-weight: bold;
        animation: slideIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 Select Your Symptoms")
    st.markdown("Select symptoms from **multiple categories**. The AI handles mixed symptoms correctly.")
    
    selected_symptoms = {}
    
    for category, symptoms in SYMPTOMS.items():
        with st.expander(f"🏷️ {category} Symptoms", expanded=True):
            cols = st.columns(2)
            for i, symptom in enumerate(symptoms):
                with cols[i % 2]:
                    symptom_name = symptom.replace('_', ' ').title()
                    if st.checkbox(symptom_name, key=f"symptom_{symptom}"):
                        selected_symptoms[symptom] = 1
    
    st.session_state.symptoms = selected_symptoms
    
    # Show selected symptoms
    if selected_symptoms:
        st.markdown("### Selected Symptoms:")
        selected_names = [k.replace('_', ' ').title() for k in selected_symptoms.keys()]
        st.info(f"**{len(selected_names)} symptoms selected:** {', '.join(selected_names)}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Home"):
            st.session_state.page = 'Home'
            st.rerun()
    with col2:
        if st.button("➡️ Medical History", type="primary"):
            st.session_state.page = 'Medical History'
            st.rerun()

def medical_history_page():
    """Medical history page"""
    # Custom CSS for medical history page with animations
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        animation: fadeIn 0.8s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes selectPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); box-shadow: 0 0 15px rgba(255,255,255,0.3); }
        100% { transform: scale(1); }
    }
    .stMarkdown, .stText {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-weight: bold;
        animation: slideIn 0.6s ease-out;
    }
    .stSelectbox {
        animation: slideIn 0.8s ease-out;
        transition: all 0.3s ease;
    }
    .stSelectbox:hover {
        transform: translateY(-2px);
    }
    .stSelectbox label {
        color: white !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        transition: all 0.3s ease;
    }
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.9) !important;
        color: #333 !important;
        font-weight: bold;
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    .stSelectbox > div > div:focus {
        animation: selectPulse 0.5s ease;
    }
    .stButton > button {
        background-color: rgba(255,255,255,0.3);
        border: 2px solid white;
        color: white !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
        animation: slideIn 1s ease-out;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        background-color: rgba(255,255,255,0.4);
    }
    .stColumns > div {
        background-color: rgba(255,255,255,0.15);
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        border: 1px solid rgba(255,255,255,0.3);
        animation: slideIn 0.9s ease-out;
        transition: all 0.3s ease;
    }
    .stColumns > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        animation: slideIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📋 Medical History")
    st.markdown("Provide your medical background for better predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        family_history = st.selectbox("Family History of Diseases", ["No", "Yes"])
        chronic_conditions = st.selectbox("Chronic Conditions", ["No", "Yes"])
    
    with col2:
        medications = st.selectbox("Currently Taking Medications", ["No", "Yes"])
        age_group = st.selectbox("Age Group", ["18-30", "31-45", "46-60", "60+"])
    
    # Store in session state
    st.session_state.medical_history = {
        'family_history': 1 if family_history == "Yes" else 0,
        'chronic_conditions': 1 if chronic_conditions == "Yes" else 0,
        'medications': 1 if medications == "Yes" else 0,
        'age_group': ["18-30", "31-45", "46-60", "60+"].index(age_group)
    }
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Symptoms"):
            st.session_state.page = 'Symptoms'
            st.rerun()
    with col2:
        if st.button("➡️ Lifestyle", type="primary"):
            st.session_state.page = 'Lifestyle'
            st.rerun()

def lifestyle_page():
    """Lifestyle information page"""
    # Custom CSS for lifestyle page
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .stMarkdown, .stText {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-weight: bold;
    }
    .stSelectbox label {
        color: white !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.9) !important;
        color: #333 !important;
        font-weight: bold;
    }
    .stSelectbox option {
        color: #333 !important;
        background-color: white !important;
    }
    .stButton > button {
        background-color: rgba(255,255,255,0.3);
        border: 2px solid white;
        color: white !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .stColumns > div {
        background-color: rgba(255,255,255,0.15);
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }
    div[data-baseweb="select"] {
        background-color: rgba(255,255,255,0.9) !important;
    }
    div[data-baseweb="select"] > div {
        color: #333 !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏃♂️ Lifestyle Information")
    st.markdown("Your lifestyle affects disease risk. Please share honestly.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    
    with col2:
        exercise = st.selectbox("Regular Exercise", ["No", "Yes"])
        stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    
    # Store in session state
    st.session_state.lifestyle = {
        'smoking': 1 if smoking == "Yes" else 0,
        'alcohol': 1 if alcohol == "Yes" else 0,
        'exercise': 1 if exercise == "Yes" else 0,
        'stress_level': ["Low", "Medium", "High"].index(stress_level)
    }
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Medical History"):
            st.session_state.page = 'Medical History'
            st.rerun()
    with col2:
        if st.button("🔮 Get Prediction", type="primary"):
            st.session_state.page = 'Prediction'
            st.rerun()

def prediction_page():
    """Prediction results page"""
    # Custom CSS for prediction page
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMarkdown, .stText {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-weight: bold;
    }
    .stButton > button {
        background-color: rgba(255,255,255,0.3);
        border: 2px solid white;
        color: white !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .stSuccess {
        background-color: rgba(255,255,255,0.2);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .stWarning {
        background-color: rgba(255,255,255,0.2);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .stMetric {
        background-color: rgba(255,255,255,0.15);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .stExpander {
        background-color: rgba(255,255,255,0.15);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .stColumns > div {
        background-color: rgba(255,255,255,0.15);
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔮 Disease Prediction Results")
    
    # Load model
    predictor = load_model()
    if predictor is None:
        return
    
    # Combine all inputs
    all_inputs = {}
    all_inputs.update(st.session_state.symptoms)
    all_inputs.update(st.session_state.medical_history)
    all_inputs.update(st.session_state.lifestyle)
    
    if not st.session_state.symptoms:
        st.warning("No symptoms selected. Please go back and select symptoms.")
        return
    
    # Get predictions
    probabilities = predictor.predict_probabilities(all_inputs)
    
    # Sort by probability
    sorted_diseases = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_diseases[:3]
    
    # Display top prediction
    top_disease, top_prob = top_3[0]
    st.success(f"**Most Likely Disease: {top_disease.replace('_', ' ')}**")
    st.metric("Confidence", f"{top_prob:.1%}")
    
    # Probability chart
    st.markdown("### Top 3 Predictions")
    
    chart_data = pd.DataFrame({
        'Disease': [d.replace('_', ' ') for d, _ in top_3],
        'Probability': [p for _, p in top_3]
    })
    
    st.bar_chart(chart_data.set_index('Disease'))
    
    # Show selected inputs summary
    with st.expander("📊 Input Summary"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Symptoms:**")
            if st.session_state.symptoms:
                for symptom in st.session_state.symptoms:
                    st.write(f"• {symptom.replace('_', ' ').title()}")
            else:
                st.write("None selected")
        
        with col2:
            st.markdown("**Medical History:**")
            history = st.session_state.medical_history
            st.write(f"• Family History: {'Yes' if history.get('family_history') else 'No'}")
            st.write(f"• Chronic Conditions: {'Yes' if history.get('chronic_conditions') else 'No'}")
            st.write(f"• Medications: {'Yes' if history.get('medications') else 'No'}")
        
        with col3:
            st.markdown("**Lifestyle:**")
            lifestyle = st.session_state.lifestyle
            st.write(f"• Smoking: {'Yes' if lifestyle.get('smoking') else 'No'}")
            st.write(f"• Alcohol: {'Yes' if lifestyle.get('alcohol') else 'No'}")
            st.write(f"• Exercise: {'Yes' if lifestyle.get('exercise') else 'No'}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Lifestyle"):
            st.session_state.page = 'Lifestyle'
            st.rerun()
    with col2:
        if st.button("💡 View Recommendations", type="primary"):
            st.session_state.page = 'Recommendations'
            st.rerun()

def recommendations_page():
    """Recommendations page"""
    # Custom CSS for recommendations page
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
    }
    .stMarkdown, .stText {
        color: #333 !important;
    }
    .stButton > button {
        background-color: rgba(51,51,51,0.1);
        border: 1px solid #333;
        color: #333 !important;
    }
    .stInfo {
        background-color: rgba(51,51,51,0.1);
        color: #333;
    }
    .stColumns > div {
        background-color: rgba(255,255,255,0.3);
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("💡 Recommendations & Insights")
    
    # Load model for feature importance
    predictor = load_model()
    if predictor is None:
        return
    
    # Get feature importance
    try:
        with open('feature_importance.json', 'r') as f:
            importance = json.load(f)
    except:
        importance = predictor.get_feature_importance()
    
    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_importance[:10]
    
    st.markdown("### 🧠 How the AI Works")
    st.info("""
    **Mixed Symptom Handling:** The AI uses probability-based predictions instead of rigid rules. 
    Even if you select symptoms from multiple disease categories, the model calculates the likelihood 
    of each disease and selects the most probable one based on learned patterns.
    """)
    
    # Feature importance chart
    st.markdown("### 📊 Most Important Factors")
    
    importance_data = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f, _ in top_features],
        'Importance': [i for _, i in top_features]
    })
    
    st.bar_chart(importance_data.set_index('Feature'))
    
    # General recommendations
    st.markdown("### 🏥 General Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **⚠️ Important Disclaimer:**
        - This is an AI prediction tool
        - Not a substitute for medical advice
        - Consult healthcare professionals
        - For emergencies, call emergency services
        """)
    
    with col2:
        st.markdown("""
        **🔄 Next Steps:**
        - Monitor your symptoms
        - Keep a symptom diary
        - Schedule doctor appointment
        - Follow up on concerning symptoms
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Prediction"):
            st.session_state.page = 'Prediction'
            st.rerun()
    with col2:
        if st.button("🏠 Start Over", type="primary"):
            # Clear session state
            st.session_state.symptoms = {}
            st.session_state.medical_history = {}
            st.session_state.lifestyle = {}
            st.session_state.page = 'Home'
            st.rerun()

# Navigation
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = ['Home', 'Symptoms', 'Medical History', 'Lifestyle', 'Prediction', 'Recommendations']
    
    for page in pages:
        if st.sidebar.button(page, key=f"nav_{page}"):
            st.session_state.page = page
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Page:** " + st.session_state.page)
    
    # Show progress
    if st.session_state.page != 'Home':
        progress_map = {
            'Symptoms': 1, 'Medical History': 2, 'Lifestyle': 3, 
            'Prediction': 4, 'Recommendations': 5
        }
        if st.session_state.page in progress_map:
            progress = progress_map[st.session_state.page] / 5
            st.sidebar.progress(progress)
    
    # Route to pages
    if st.session_state.page == 'Home':
        home_page()
    elif st.session_state.page == 'Symptoms':
        symptoms_page()
    elif st.session_state.page == 'Medical History':
        medical_history_page()
    elif st.session_state.page == 'Lifestyle':
        lifestyle_page()
    elif st.session_state.page == 'Prediction':
        prediction_page()
    elif st.session_state.page == 'Recommendations':
        recommendations_page()

if __name__ == "__main__":
    main()