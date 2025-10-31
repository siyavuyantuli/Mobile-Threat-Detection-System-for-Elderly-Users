# mobile_threat_detector_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mobile Threat Detector",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff4b4b;
        font-weight: bold;
    }
    .section-header {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    try:
        # Load deployment package
        deployment_package = joblib.load('mobile_threat_detection_model.pkl')
        
        # Validate the loaded data
        required_keys = ['best_model', 'feature_columns']
        for key in required_keys:
            if key not in deployment_package:
                st.error(f"❌ Missing key in model data: {key}")
                return None, None, None, None
                
        st.success(f"✅ Model loaded: {deployment_package.get('best_model_name', 'Unknown')}")

        # Load individual models
        rf_model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')

        # REMOVED LSTM loading since we don't have TensorFlow
        lstm_model = None

        return deployment_package, rf_model, scaler, lstm_model

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def calculate_fallback_risk(user_data):
    """Calculate risk based on rules when AI model fails - FIXED VERSION"""
    st.info("🔄 Using rule-based risk assessment")
    
    risk_score = 0
    
    # HIGH-RISK FACTORS (Major security threats)
    # Suspicious SMS (phishing attempts)
    risk_score += min(user_data.get('suspicious_sms_count', 0) * 8, 40)
    
    # Failed login attempts (brute force attacks)
    risk_score += min(user_data.get('failed_login_attempts', 0) * 7, 35)
    
    # Unknown network connections (potential MITM attacks)
    risk_score += min(user_data.get('unknown_network_connections', 0) * 6, 30)
    
    # Outdated apps (security vulnerabilities)
    risk_score += min(user_data.get('outdated_apps_count', 0) * 3, 25)
    
    # Public WiFi usage (unsecured networks)
    risk_score += min(user_data.get('public_wifi_usage_hours', 0) * 2, 20)
    
    # MEDIUM-RISK FACTORS (Security hygiene)
    # No security training
    if not user_data.get('security_training_completed', 0):
        risk_score += 15
    
    # Old OS updates
    if user_data.get('days_since_os_update', 0) > 90:
        risk_score += 10
    
    # High permission risks
    if user_data.get('permission_risk_index', 0) > 15:
        risk_score += 12
    
    # No VPN usage
    if not user_data.get('vpn_usage', 0):
        risk_score += 8
        
    # LOW-RISK FACTORS (Behavioral patterns)
    # Low tech literacy
    if user_data.get('tech_literacy_level', 3) < 2:
        risk_score += 5
    
    # Advanced age
    if user_data.get('age', 65) > 75:
        risk_score += 3
        
    # Threat history
    risk_score += min(user_data.get('threat_severity', 0) * 2, 10)
    
    # Spike detections
    if user_data.get('screen_time_spike', 0):
        risk_score += 3
    if user_data.get('data_usage_spike', 0):
        risk_score += 4
    if user_data.get('abnormal_battery_drain', 0):
        risk_score += 3
    
    # Cap at 100%
    risk_score = min(risk_score, 100)
    
    # Determine threat (threshold at 40% risk) - FIXED LOGIC
    prediction = 1 if risk_score >= 40 else 0
    
    # Calculate confidence based on how clear the risk assessment is
    if risk_score >= 70 or risk_score <= 20:
        confidence = 0.95  # Very clear cases - high confidence
    elif risk_score >= 55 or risk_score <= 35:
        confidence = 0.85  # Moderately clear cases
    else:
        confidence = 0.75  # Borderline cases - lower confidence
    
    st.write(f"🔧 Rule-based risk score: {risk_score}%")
    st.write(f"🔧 Rule-based prediction: {'THREAT' if prediction == 1 else 'NO THREAT'}")
    st.write(f"🔧 Rule-based confidence: {confidence:.1%}")
    
    # FIXED: Proper probability calculation
    # If we predict THREAT, probability should be high when risk is high
    # If we predict NO THREAT, probability should be high when risk is low
    if prediction == 1:
        # For threat predictions: higher risk = higher probability
        rule_probability = max(0.5, risk_score / 100.0)  # At least 50% probability for threats
    else:
        # For no-threat predictions: lower risk = higher probability  
        rule_probability = max(0.5, (100 - risk_score) / 100.0)  # At least 50% probability for no-threat
    
    return prediction, confidence, rule_probability

def predict_threat_ai(user_data, model_data):
    """Make prediction using AI model with comprehensive debugging"""
    try:
        st.write("🔍 **AI MODEL DEBUG MODE**")
        
        # Check model data structure
        st.write("### Model Data Check:")
        st.write(f"- Model keys: {list(model_data.keys())}")
        st.write(f"- Best model name: {model_data.get('best_model_name', 'Not found')}")
        st.write(f"- Best model type: {type(model_data.get('best_model'))}")
        st.write(f"- Feature columns: {len(model_data.get('feature_columns', []))} features")
        
        # Extract components
        model = model_data.get('best_model')
        scaler = model_data.get('scaler')
        feature_columns = model_data.get('feature_columns', [])
        
        if model is None:
            st.error("❌ MODEL IS NONE - Using fallback calculation")
            return 0, 0.0
            
        # Show what features the model expects vs what we're providing
        st.write("### Feature Analysis:")
        st.write(f"Model expects: {len(feature_columns)} features")
        st.write(f"We're providing: {len(user_data)} input values")
        
        # Check for missing features
        missing_features = [f for f in feature_columns if f not in user_data]
        if missing_features:
            st.error(f"❌ CRITICAL: {len(missing_features)} features missing from input!")
            st.write("Missing features:", missing_features)
            return 0, 0.0
        
        # Create input dataframe with ALL expected features
        input_df = pd.DataFrame(columns=feature_columns)
        for col in feature_columns:
            if col in user_data:
                input_df[col] = [user_data[col]]
            else:
                st.error(f"❌ Feature {col} not found in user data")
                input_df[col] = [0]  # Emergency fallback
        
        st.write(f"✅ Final input shape: {input_df.shape}")
        st.write(f"✅ All {len(feature_columns)} features available")
        
        # Scale features and predict
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            st.warning("❌ No scaler found!")
            input_scaled = input_df.values
            
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_scaled)[0][1]
            prediction = model.predict(input_scaled)[0]
            st.write(f"🎯 Raw probability (threat): {probability:.4f}")
            st.write(f"🎯 Binary prediction: {prediction}")
        else:
            prediction = model.predict(input_scaled)[0]
            probability = float(prediction)  # For models without probability
            st.write(f"🎯 Direct prediction: {prediction}")
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"❌ AI Prediction failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return 0, 0.0

def calculate_ai_confidence(ai_probability):
    """Calculate true AI confidence (distance from decision boundary)"""
    # Confidence = how far from 0.5 (the decision boundary)
    # Converts to 0-1 scale where 1.0 = maximum confidence
    confidence = abs(ai_probability - 0.5) * 2
    return confidence

def get_confidence_level(confidence):
    """Get confidence level description"""
    if confidence >= 0.8:
        return "high", "confidence-high"
    elif confidence >= 0.6:
        return "medium", "confidence-medium"
    else:
        return "low", "confidence-low"

def predict_threat(user_data, model_data):
    """Make prediction using AI model with PROPER confidence handling"""
    # Try AI model first
    if model_data and model_data.get('best_model') is not None:
        ai_prediction, ai_probability = predict_threat_ai(user_data, model_data)
        
        # PROPER CONFIDENCE CALCULATION
        ai_confidence = calculate_ai_confidence(ai_probability)
        confidence_level, confidence_class = get_confidence_level(ai_confidence)
        
        st.write(f"🤖 AI Raw Probability: {ai_probability:.3f}")
        st.write(f"🤖 AI Confidence: <span class='{confidence_class}'>{ai_confidence:.1%} ({confidence_level})</span>", unsafe_allow_html=True)
        st.write(f"🤖 AI Prediction: {'THREAT' if ai_prediction == 1 else 'NO THREAT'}")
        
        # Only use rule-based if AI is truly uncertain (low confidence)
        if ai_confidence < 0.6:  # Less than 60% confident
            st.warning(f"🤖 AI has {confidence_level} confidence, using rule-based assessment")
            rule_prediction, rule_confidence, rule_probability = calculate_fallback_risk(user_data)
            return rule_prediction, rule_probability, "rule-based"
        else:
            # Use AI results with high/medium confidence
            st.success(f"🤖 AI has {confidence_level} confidence, using AI assessment")
            return ai_prediction, ai_probability, "ai"
        
    else:
        st.warning("🤖 AI model not available, using rule-based assessment")
        rule_prediction, rule_confidence, rule_probability = calculate_fallback_risk(user_data)
        return rule_prediction, rule_probability, "rule-based"

def calculate_risk_score(prediction, probability, user_data):
    """Calculate comprehensive risk score - COMPLETELY FIXED VERSION"""
    
    # BASE RISK: Use the probability directly
    base_risk = probability * 100
    
    # For THREAT predictions, we should have higher risk scores
    # For NO THREAT predictions, we should have lower risk scores
    if prediction == 1:
        # This is a THREAT - risk should be at least moderate
        base_risk = max(base_risk, 40)  # At least 40% risk for threats
    else:
        # This is NO THREAT - risk should be capped lower
        base_risk = min(base_risk, 60)  # At most 60% risk for no-threats
    
    # Additional risk factors (much smaller adjustment)
    risk_factors = {
        'suspicious_sms_count': 1,
        'failed_login_attempts': 2, 
        'outdated_apps_count': 0.5,
        'public_wifi_usage_hours': 0.5,
        'unknown_network_connections': 1.5
    }

    additional_risk = 0
    for factor, weight in risk_factors.items():
        if factor in user_data:
            additional_risk += user_data[factor] * weight

    # Cap additional risk at 10%
    additional_risk = min(additional_risk, 10)
    
    # Calculate total risk
    if prediction == 1:
        # For threats: base risk + additional risk
        total_risk = min(base_risk + additional_risk, 100)
    else:
        # For no-threats: base risk - additional risk (but not below 0)
        total_risk = max(base_risk - additional_risk, 0)
    
    return total_risk

def main():
    # Header
    st.markdown('<h1 class="main-header">📱 Mobile Threat Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### For Elderly Users - AI-Powered Security Protection")

    # Load models with debug info
    with st.spinner('Loading AI models...'):
        models = load_models()
    
    # DEBUG: Show what models loaded
    if models[0] is not None:
        deployment_package, rf_model, scaler, lstm_model = models
        st.success(f"✅ AI Model Loaded: {deployment_package.get('best_model_name', 'Random Forest')}")
        st.write(f"🔧 Model Type: {type(deployment_package.get('best_model'))}")
        st.write(f"🔧 Feature Count: {len(deployment_package.get('feature_columns', []))}")
    else:
        st.error("❌ AI Models Failed to Load - Using Rule-Based Mode Only")
        deployment_package = {'feature_columns': []}

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Single User Analysis", "Model Performance", "About"]
    )

    if app_mode == "Single User Analysis":
        single_user_analysis(deployment_package)
    elif app_mode == "Model Performance":
        model_performance(deployment_package)
    elif app_mode == "About":
        about_section()

def single_user_analysis(deployment_package):
    """Single user threat analysis interface with ALL 53 features"""

    st.header("🔍 Single User Threat Analysis")
    st.info("📋 Please provide all 53 features for accurate threat detection")

    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 User Profile", 
        "⚙️ Security Settings", 
        "📊 Usage Patterns", 
        "⚠️ Risk Factors", 
        "🔧 Advanced Metrics"
    ])

    user_data = {}

    with tab1:
        st.markdown('<div class="section-header">👤 User Profile & Demographics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 60, 100, 75)
            tech_literacy = st.select_slider(
                "Technology Literacy Level",
                options=['very_low', 'low', 'medium', 'high', 'very_high'],
                value='medium'
            )
            gender = st.radio("Gender", ['Male', 'Female'])
            
        with col2:
            device_type = st.selectbox("Device Type", ['Android', 'iOS'])
            location = st.selectbox("Location Type", ['Home', 'Work', 'Public', 'Traveling'])
            time_of_day = st.slider("Current Time of Day (24h)", 0, 23, 12)

        # Convert categorical to numerical
        tech_literacy_map = {'very_low': 1, 'low': 2, 'medium': 3, 'high': 4, 'very_high': 5}
        location_map = {'Home': 1, 'Work': 2, 'Public': 3, 'Traveling': 4}
        
        user_data.update({
            'age': age,
            'tech_literacy_level': tech_literacy_map[tech_literacy],
            'gender': 1 if gender == 'Male' else 0,
            'device_type': 1 if device_type == 'Android' else 0,
            'location': location_map[location],
            'time_of_day': time_of_day
        })

    with tab2:
        st.markdown('<div class="section-header">⚙️ Security Settings & Configuration</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            security_training = st.checkbox("Completed Security Training")
            vpn_usage = st.checkbox("Uses VPN Regularly")
            os_version_outdated = st.checkbox("OS Version is Outdated")
            apps_from_unknown_sources = st.checkbox("Apps from Unknown Sources")
            
        with col2:
            days_since_update = st.slider("Days Since OS Update", 0, 365, 30)
            outdated_apps = st.slider("Outdated Apps Count", 0, 20, 2)
            previous_incidents = st.slider("Previous Security Incidents", 0, 10, 0)

        user_data.update({
            'security_training_completed': 1 if security_training else 0,
            'days_since_os_update': days_since_update,
            'outdated_apps_count': outdated_apps,
            'vpn_usage': 1 if vpn_usage else 0,
            'previous_incidents': previous_incidents,
            'os_version_outdated': 1 if os_version_outdated else 0,
            'apps_from_unknown_sources': 1 if apps_from_unknown_sources else 0
        })

    with tab3:
        st.markdown('<div class="section-header">📊 Usage Patterns & Behavior</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            daily_usage = st.slider("Daily App Usage (minutes)", 0, 480, 120)
            data_usage = st.slider("Daily Data Usage (MB)", 0, 1000, 200)
            battery_drain = st.slider("Daily Battery Drain (%)", 0, 100, 30)
            late_night_usage = st.slider("Late Night Usage (minutes)", 0, 240, 0)
            
        with col2:
            wifi_sessions = st.slider("Daily WiFi Sessions", 0, 20, 5)
            mobile_sessions = st.slider("Daily Mobile Data Sessions", 0, 20, 3)
            background_data = st.slider("Background Data Usage (MB)", 0, 500, 50)
            weekly_variance = st.slider("Weekly Usage Variance", 0, 100, 20)

        # Delta calculations
        delta_battery = st.slider("Battery Drain Change vs Last Week (%)", -50, 50, 0)
        delta_data = st.slider("Data Usage Change vs Last Week (MB)", -500, 500, 0)
        delta_app_usage = st.slider("App Usage Change vs Last Week (min)", -240, 240, 0)

        user_data.update({
            'daily_app_usage_minutes': daily_usage,
            'daily_data_usage_mb': data_usage,
            'daily_battery_drain_pct': battery_drain,
            'late_night_usage_minutes': late_night_usage,
            'wifi_sessions_daily': wifi_sessions,
            'mobile_data_sessions_daily': mobile_sessions,
            'background_data_usage_mb': background_data,
            'weekly_usage_variance': weekly_variance,
            'delta_battery_drain_pct': delta_battery,
            'delta_data_usage_mb': delta_data,
            'delta_app_usage_minutes': delta_app_usage
        })

    with tab4:
        st.markdown('<div class="section-header">⚠️ Immediate Risk Factors</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            suspicious_sms = st.slider("Suspicious SMS Count", 0, 10, 0)
            failed_logins = st.slider("Failed Login Attempts", 0, 10, 0)
            public_wifi = st.slider("Public WiFi Usage (hours)", 0, 24, 2)
            unknown_networks = st.slider("Unknown Network Connections", 0, 10, 0)
            
        with col2:
            suspicious_calls = st.slider("Suspicious Call Count", 0, 10, 0)
            app_crashes = st.slider("Daily App Crash Count", 0, 10, 0)
            connection_changes = st.slider("Connection Type Changes", 0, 10, 2)
            unusual_time_activity = st.slider("Unusual Time Activity Score", 0, 10, 0)

        user_data.update({
            'suspicious_sms_count': suspicious_sms,
            'failed_login_attempts': failed_logins,
            'public_wifi_usage_hours': public_wifi,
            'unknown_network_connections': unknown_networks,
            'suspicious_call_count': suspicious_calls,
            'app_crash_count_daily': app_crashes,
            'connection_type_changes': connection_changes,
            'unusual_time_activity': unusual_time_activity
        })

    with tab5:
        st.markdown('<div class="section-header">🔧 Advanced Security Metrics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            threat_severity = st.slider("Previous Threat Severity", 0, 10, 0)
            camera_permissions = st.slider("Camera Permission Apps", 0, 10, 2)
            location_permissions = st.slider("Location Permission Apps", 0, 10, 3)
            sms_permissions = st.slider("SMS Permission Apps", 0, 10, 1)
            contacts_permissions = st.slider("Contacts Permission Apps", 0, 10, 2)
            
        with col2:
            new_apps_installed = st.slider("New Apps Installed This Week", 0, 10, 1)
            financial_loss = st.slider("Financial Loss Indicator", 0, 10, 0)
            data_compromised = st.slider("Data Compromised Indicator", 0, 10, 0)
            response_time = st.slider("Response Time (seconds)", 0, 3600, 300)
            network_type = st.selectbox("Primary Network Type", ['Home_WiFi', 'Public_WiFi', 'Mobile_Data', 'Unknown'])

        # Additional advanced metrics
        col3, col4 = st.columns(2)
        with col3:
            screen_time_spike = st.checkbox("Screen Time Spike Detected")
            data_usage_spike = st.checkbox("Data Usage Spike Detected")
            abnormal_battery = st.checkbox("Abnormal Battery Drain")
            
        with col4:
            threat_type = st.selectbox("Threat Type Category", [
                'None', 'Phishing', 'Malware', 'Data_Theft', 'Financial_Fraud', 
                'Identity_Theft', 'Ransomware', 'Other'
            ])
            app_name = st.selectbox("Primary App Concern", [
                'None', 'Browser', 'Social_Media', 'Banking', 'Email', 
                'Messaging', 'Games', 'Other'
            ])

        # Convert categorical to numerical
        threat_type_map = {
            'None': 0, 'Phishing': 1, 'Malware': 2, 'Data_Theft': 3, 
            'Financial_Fraud': 4, 'Identity_Theft': 5, 'Ransomware': 6, 'Other': 7
        }
        app_name_map = {
            'None': 0, 'Browser': 1, 'Social_Media': 2, 'Banking': 3, 
            'Email': 4, 'Messaging': 5, 'Games': 6, 'Other': 7
        }
        network_type_map = {
            'Home_WiFi': 1, 'Public_WiFi': 2, 'Mobile_Data': 3, 'Unknown': 4
        }

        # Calculate composite scores
        permission_risk_index = camera_permissions + location_permissions + sms_permissions + contacts_permissions
        abnormal_behavior_score = min(100, (suspicious_sms * 10 + failed_logins * 8 + unknown_networks * 6))
        network_security_index = 100 - (public_wifi * 3 + unknown_networks * 5)
        battery_data_ratio = battery_drain / max(data_usage, 1)
        usage_efficiency = max(0, 100 - (battery_drain * 0.5 + data_usage * 0.1))

        user_data.update({
            'threat_severity': threat_severity,
            'camera_permission_apps': camera_permissions,
            'location_permission_apps': location_permissions,
            'sms_permission_apps': sms_permissions,
            'contacts_permission_apps': contacts_permissions,
            'new_apps_installed_week': new_apps_installed,
            'financial_loss': financial_loss,
            'data_compromised': data_compromised,
            'response_time_seconds': response_time,
            'network_type': network_type_map[network_type],
            'screen_time_spike': 1 if screen_time_spike else 0,
            'data_usage_spike': 1 if data_usage_spike else 0,
            'abnormal_battery_drain': 1 if abnormal_battery else 0,
            'threat_type': threat_type_map[threat_type],
            'app_name': app_name_map[app_name],
            'permission_risk_index': permission_risk_index,
            'abnormal_behavior_score': abnormal_behavior_score,
            'network_security_index': network_security_index,
            'battery_data_ratio': battery_data_ratio,
            'usage_efficiency': usage_efficiency
        })

    # Display feature count validation
    st.info(f"📊 Total features provided: {len(user_data)}/53")
    if len(user_data) < 53:
        st.warning(f"❌ Missing {53 - len(user_data)} features. Please check all tabs.")
    else:
        st.success("✅ All 53 features provided!")

    # Analyze button
    if st.button("🔍 Analyze Threat Level", type="primary"):
        if len(user_data) < 53:
            st.error("❌ Please provide all 53 features for accurate analysis")
            return
            
        with st.spinner('Analyzing user data with all 53 features...'):
            # Use the new combined prediction system
            main_prediction, main_probability, system_used = predict_threat(user_data, deployment_package)

            # Calculate risk score - NOW USING PREDICTION + PROBABILITY
            risk_score = calculate_risk_score(main_prediction, main_probability, user_data)

            # Display results
            st.header("📊 Analysis Results")

            # System used indicator
            system_color = "🟢" if system_used == "ai" else "🟡"
            st.write(f"{system_color} **System Used:** {system_used.upper()}")

            # Risk level display
            col1, col2, col3 = st.columns(3)

            with col1:
                if risk_score >= 70:
                    st.markdown(f'<div class="risk-high">HIGH RISK: {risk_score:.1f}%</div>', unsafe_allow_html=True)
                elif risk_score >= 30:
                    st.markdown(f'<div class="risk-medium">MEDIUM RISK: {risk_score:.1f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">LOW RISK: {risk_score:.1f}%</div>', unsafe_allow_html=True)

            with col2:
                threat_status = "🚨 THREAT DETECTED" if main_prediction == 1 else "✅ NO THREAT"
                st.metric("Threat Status", threat_status)

            with col3:
                confidence_display = main_probability * 100
                st.metric("Probability", f"{confidence_display:.1f}%")

            # Debug information
            with st.expander("🔍 Technical Details"):
                st.write(f"Final Prediction: {main_prediction}")
                st.write(f"Final Probability: {main_probability:.3f}")
                st.write(f"Calculated Risk Score: {risk_score:.1f}%")
                st.write(f"System Used: {system_used}")
                st.write(f"Total Features Provided: {len(user_data)}")
                
                # Show which system was more confident
                if system_used == "ai":
                    ai_confidence = calculate_ai_confidence(main_probability)
                    st.write(f"AI Confidence Level: {ai_confidence:.1%}")

            # Risk factors breakdown
            st.subheader("🔍 Risk Factors Breakdown")

            risk_factors = [
                ("Suspicious SMS", user_data['suspicious_sms_count'] * 8),
                ("Failed Logins", user_data['failed_login_attempts'] * 7),
                ("Unknown Networks", user_data['unknown_network_connections'] * 6),
                ("Outdated Apps", user_data['outdated_apps_count'] * 3),
                ("Public WiFi", user_data['public_wifi_usage_hours'] * 2),
                ("No Security Training", 15 if not user_data.get('security_training_completed', 0) else 0),
                ("Old OS", 10 if user_data.get('days_since_os_update', 0) > 90 else 0),
                ("Permission Risk", user_data['permission_risk_index'] * 0.5),
                ("Suspicious Calls", user_data['suspicious_call_count'] * 3),
                ("App Crashes", user_data['app_crash_count_daily'] * 2)
            ]

            risk_df = pd.DataFrame(risk_factors, columns=['Factor', 'Risk Score'])
            risk_df = risk_df[risk_df['Risk Score'] > 0].sort_values('Risk Score', ascending=False)

            if not risk_df.empty:
                fig = px.bar(risk_df, x='Risk Score', y='Factor', orientation='h',
                            title="Contributing Risk Factors")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant risk factors detected.")

            # Recommendations
            st.subheader("💡 Security Recommendations")

            recommendations = []
            if user_data['outdated_apps_count'] > 5:
                recommendations.append("🔸 **URGENT**: Update outdated applications immediately")
            if user_data['public_wifi_usage_hours'] > 5:
                recommendations.append("🔸 **URGENT**: Avoid public WiFi or use VPN")
            if user_data['failed_login_attempts'] > 0:
                recommendations.append("🔸 **URGENT**: Change passwords and enable 2FA")
            if user_data['suspicious_sms_count'] > 0:
                recommendations.append("🔸 **URGENT**: Do not click suspicious links")
            if user_data['permission_risk_index'] > 15:
                recommendations.append("🔸 **HIGH**: Review and reduce app permissions")
            if not user_data.get('security_training_completed', 0):
                recommendations.append("🔸 **HIGH**: Complete security awareness training")
            if user_data['days_since_os_update'] > 90:
                recommendations.append("🔸 **HIGH**: Update device operating system")
            if not user_data.get('vpn_usage', 0):
                recommendations.append("🔸 **MEDIUM**: Use VPN on public networks")
            if user_data['apps_from_unknown_sources']:
                recommendations.append("🔸 **MEDIUM**: Only install apps from official stores")

            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("All security practices look good!")

def single_user_analysis(deployment_package):
    """Single user threat analysis interface with ALL 53 features"""

    st.header("🔍 Single User Threat Analysis")
    st.info("📋 Please provide all 53 features for accurate threat detection")

    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 User Profile", 
        "⚙️ Security Settings", 
        "📊 Usage Patterns", 
        "⚠️ Risk Factors", 
        "🔧 Advanced Metrics"
    ])

    user_data = {}

    with tab1:
        st.markdown('<div class="section-header">👤 User Profile & Demographics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 60, 100, 75)
            tech_literacy = st.select_slider(
                "Technology Literacy Level",
                options=['very_low', 'low', 'medium', 'high', 'very_high'],
                value='medium'
            )
            gender = st.radio("Gender", ['Male', 'Female'])
            
        with col2:
            device_type = st.selectbox("Device Type", ['Android', 'iOS'])
            location = st.selectbox("Location Type", ['Home', 'Work', 'Public', 'Traveling'])
            time_of_day = st.slider("Current Time of Day (24h)", 0, 23, 12)

        # Convert categorical to numerical
        tech_literacy_map = {'very_low': 1, 'low': 2, 'medium': 3, 'high': 4, 'very_high': 5}
        location_map = {'Home': 1, 'Work': 2, 'Public': 3, 'Traveling': 4}
        
        user_data.update({
            'age': age,
            'tech_literacy_level': tech_literacy_map[tech_literacy],
            'gender': 1 if gender == 'Male' else 0,
            'device_type': 1 if device_type == 'Android' else 0,
            'location': location_map[location],
            'time_of_day': time_of_day
        })

    with tab2:
        st.markdown('<div class="section-header">⚙️ Security Settings & Configuration</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            security_training = st.checkbox("Completed Security Training")
            vpn_usage = st.checkbox("Uses VPN Regularly")
            os_version_outdated = st.checkbox("OS Version is Outdated")
            apps_from_unknown_sources = st.checkbox("Apps from Unknown Sources")
            
        with col2:
            days_since_update = st.slider("Days Since OS Update", 0, 365, 30)
            outdated_apps = st.slider("Outdated Apps Count", 0, 20, 2)
            previous_incidents = st.slider("Previous Security Incidents", 0, 10, 0)

        user_data.update({
            'security_training_completed': 1 if security_training else 0,
            'days_since_os_update': days_since_update,
            'outdated_apps_count': outdated_apps,
            'vpn_usage': 1 if vpn_usage else 0,
            'previous_incidents': previous_incidents,
            'os_version_outdated': 1 if os_version_outdated else 0,
            'apps_from_unknown_sources': 1 if apps_from_unknown_sources else 0
        })

    with tab3:
        st.markdown('<div class="section-header">📊 Usage Patterns & Behavior</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            daily_usage = st.slider("Daily App Usage (minutes)", 0, 480, 120)
            data_usage = st.slider("Daily Data Usage (MB)", 0, 1000, 200)
            battery_drain = st.slider("Daily Battery Drain (%)", 0, 100, 30)
            late_night_usage = st.slider("Late Night Usage (minutes)", 0, 240, 0)
            
        with col2:
            wifi_sessions = st.slider("Daily WiFi Sessions", 0, 20, 5)
            mobile_sessions = st.slider("Daily Mobile Data Sessions", 0, 20, 3)
            background_data = st.slider("Background Data Usage (MB)", 0, 500, 50)
            weekly_variance = st.slider("Weekly Usage Variance", 0, 100, 20)

        # Delta calculations
        delta_battery = st.slider("Battery Drain Change vs Last Week (%)", -50, 50, 0)
        delta_data = st.slider("Data Usage Change vs Last Week (MB)", -500, 500, 0)
        delta_app_usage = st.slider("App Usage Change vs Last Week (min)", -240, 240, 0)

        user_data.update({
            'daily_app_usage_minutes': daily_usage,
            'daily_data_usage_mb': data_usage,
            'daily_battery_drain_pct': battery_drain,
            'late_night_usage_minutes': late_night_usage,
            'wifi_sessions_daily': wifi_sessions,
            'mobile_data_sessions_daily': mobile_sessions,
            'background_data_usage_mb': background_data,
            'weekly_usage_variance': weekly_variance,
            'delta_battery_drain_pct': delta_battery,
            'delta_data_usage_mb': delta_data,
            'delta_app_usage_minutes': delta_app_usage
        })

    with tab4:
        st.markdown('<div class="section-header">⚠️ Immediate Risk Factors</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            suspicious_sms = st.slider("Suspicious SMS Count", 0, 10, 0)
            failed_logins = st.slider("Failed Login Attempts", 0, 10, 0)
            public_wifi = st.slider("Public WiFi Usage (hours)", 0, 24, 2)
            unknown_networks = st.slider("Unknown Network Connections", 0, 10, 0)
            
        with col2:
            suspicious_calls = st.slider("Suspicious Call Count", 0, 10, 0)
            app_crashes = st.slider("Daily App Crash Count", 0, 10, 0)
            connection_changes = st.slider("Connection Type Changes", 0, 10, 2)
            unusual_time_activity = st.slider("Unusual Time Activity Score", 0, 10, 0)

        user_data.update({
            'suspicious_sms_count': suspicious_sms,
            'failed_login_attempts': failed_logins,
            'public_wifi_usage_hours': public_wifi,
            'unknown_network_connections': unknown_networks,
            'suspicious_call_count': suspicious_calls,
            'app_crash_count_daily': app_crashes,
            'connection_type_changes': connection_changes,
            'unusual_time_activity': unusual_time_activity
        })

    with tab5:
        st.markdown('<div class="section-header">🔧 Advanced Security Metrics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            threat_severity = st.slider("Previous Threat Severity", 0, 10, 0)
            camera_permissions = st.slider("Camera Permission Apps", 0, 10, 2)
            location_permissions = st.slider("Location Permission Apps", 0, 10, 3)
            sms_permissions = st.slider("SMS Permission Apps", 0, 10, 1)
            contacts_permissions = st.slider("Contacts Permission Apps", 0, 10, 2)
            
        with col2:
            new_apps_installed = st.slider("New Apps Installed This Week", 0, 10, 1)
            financial_loss = st.slider("Financial Loss Indicator", 0, 10, 0)
            data_compromised = st.slider("Data Compromised Indicator", 0, 10, 0)
            response_time = st.slider("Response Time (seconds)", 0, 3600, 300)
            network_type = st.selectbox("Primary Network Type", ['Home_WiFi', 'Public_WiFi', 'Mobile_Data', 'Unknown'])

        # Additional advanced metrics
        col3, col4 = st.columns(2)
        with col3:
            screen_time_spike = st.checkbox("Screen Time Spike Detected")
            data_usage_spike = st.checkbox("Data Usage Spike Detected")
            abnormal_battery = st.checkbox("Abnormal Battery Drain")
            
        with col4:
            threat_type = st.selectbox("Threat Type Category", [
                'None', 'Phishing', 'Malware', 'Data_Theft', 'Financial_Fraud', 
                'Identity_Theft', 'Ransomware', 'Other'
            ])
            app_name = st.selectbox("Primary App Concern", [
                'None', 'Browser', 'Social_Media', 'Banking', 'Email', 
                'Messaging', 'Games', 'Other'
            ])

        # Convert categorical to numerical
        threat_type_map = {
            'None': 0, 'Phishing': 1, 'Malware': 2, 'Data_Theft': 3, 
            'Financial_Fraud': 4, 'Identity_Theft': 5, 'Ransomware': 6, 'Other': 7
        }
        app_name_map = {
            'None': 0, 'Browser': 1, 'Social_Media': 2, 'Banking': 3, 
            'Email': 4, 'Messaging': 5, 'Games': 6, 'Other': 7
        }
        network_type_map = {
            'Home_WiFi': 1, 'Public_WiFi': 2, 'Mobile_Data': 3, 'Unknown': 4
        }

        # Calculate composite scores
        permission_risk_index = camera_permissions + location_permissions + sms_permissions + contacts_permissions
        abnormal_behavior_score = min(100, (suspicious_sms * 10 + failed_logins * 8 + unknown_networks * 6))
        network_security_index = 100 - (public_wifi * 3 + unknown_networks * 5)
        battery_data_ratio = battery_drain / max(data_usage, 1)
        usage_efficiency = max(0, 100 - (battery_drain * 0.5 + data_usage * 0.1))

        user_data.update({
            'threat_severity': threat_severity,
            'camera_permission_apps': camera_permissions,
            'location_permission_apps': location_permissions,
            'sms_permission_apps': sms_permissions,
            'contacts_permission_apps': contacts_permissions,
            'new_apps_installed_week': new_apps_installed,
            'financial_loss': financial_loss,
            'data_compromised': data_compromised,
            'response_time_seconds': response_time,
            'network_type': network_type_map[network_type],
            'screen_time_spike': 1 if screen_time_spike else 0,
            'data_usage_spike': 1 if data_usage_spike else 0,
            'abnormal_battery_drain': 1 if abnormal_battery else 0,
            'threat_type': threat_type_map[threat_type],
            'app_name': app_name_map[app_name],
            'permission_risk_index': permission_risk_index,
            'abnormal_behavior_score': abnormal_behavior_score,
            'network_security_index': network_security_index,
            'battery_data_ratio': battery_data_ratio,
            'usage_efficiency': usage_efficiency
        })

    # DEBUG: Show exact feature comparison
    if deployment_package and 'feature_columns' in deployment_package:
        expected_features = deployment_package['feature_columns']
        provided_features = list(user_data.keys())
        
        st.markdown("---")
        st.subheader("🔍 Feature Validation Debug")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Expected Features (53):**")
            for i, feature in enumerate(expected_features, 1):
                status = "✅" if feature in provided_features else "❌"
                st.write(f"{i:2d}. {status} {feature}")
        
        with col2:
            st.write("**Provided Features:**")
            for i, feature in enumerate(provided_features, 1):
                status = "✅" if feature in expected_features else "❌"
                st.write(f"{i:2d}. {status} {feature}")
        
        # Check for mismatches
        missing_in_provided = [f for f in expected_features if f not in provided_features]
        extra_in_provided = [f for f in provided_features if f not in expected_features]
        
        if missing_in_provided:
            st.error(f"❌ Missing {len(missing_in_provided)} features: {missing_in_provided}")
        if extra_in_provided:
            st.warning(f"⚠️ Extra features provided: {extra_in_provided}")
        
        st.info(f"📊 Expected: {len(expected_features)} | Provided: {len(provided_features)} | Match: {len(set(expected_features) & set(provided_features))}")

    # Display feature count validation
    st.info(f"📊 Total features provided: {len(user_data)}")
    if deployment_package and 'feature_columns' in deployment_package:
        expected_count = len(deployment_package['feature_columns'])
        if len(user_data) < expected_count:
            st.error(f"❌ Missing {expected_count - len(user_data)} features. Please check all tabs.")
        elif len(user_data) > expected_count:
            st.warning(f"⚠️ Provided {len(user_data) - expected_count} extra features.")
        else:
            st.success(f"✅ All {expected_count} features provided!")

    # Analyze button
    if st.button("🔍 Analyze Threat Level", type="primary"):
        if deployment_package and 'feature_columns' in deployment_package:
            expected_features = deployment_package['feature_columns']
            missing_features = [f for f in expected_features if f not in user_data]
            
            if missing_features:
                st.error(f"❌ Cannot proceed - missing {len(missing_features)} features: {missing_features}")
                return
                
        with st.spinner('Analyzing user data with all features...'):
            # Use the new combined prediction system
            main_prediction, main_probability, system_used = predict_threat(user_data, deployment_package)

            # Calculate risk score - NOW USING PREDICTION + PROBABILITY
            risk_score = calculate_risk_score(main_prediction, main_probability, user_data)

            # Display results
            st.header("📊 Analysis Results")

            # System used indicator
            system_color = "🟢" if system_used == "ai" else "🟡"
            st.write(f"{system_color} **System Used:** {system_used.upper()}")

            # Risk level display
            col1, col2, col3 = st.columns(3)

            with col1:
                if risk_score >= 70:
                    st.markdown(f'<div class="risk-high">HIGH RISK: {risk_score:.1f}%</div>', unsafe_allow_html=True)
                elif risk_score >= 30:
                    st.markdown(f'<div class="risk-medium">MEDIUM RISK: {risk_score:.1f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">LOW RISK: {risk_score:.1f}%</div>', unsafe_allow_html=True)

            with col2:
                threat_status = "🚨 THREAT DETECTED" if main_prediction == 1 else "✅ NO THREAT"
                st.metric("Threat Status", threat_status)

            with col3:
                confidence_display = main_probability * 100
                st.metric("Probability", f"{confidence_display:.1f}%")

            # Debug information
            with st.expander("🔍 Technical Details"):
                st.write(f"Final Prediction: {main_prediction}")
                st.write(f"Final Probability: {main_probability:.3f}")
                st.write(f"Calculated Risk Score: {risk_score:.1f}%")
                st.write(f"System Used: {system_used}")
                st.write(f"Total Features Provided: {len(user_data)}")
                
                if deployment_package and 'feature_columns' in deployment_package:
                    st.write(f"Expected Features: {len(deployment_package['feature_columns'])}")
                    st.write(f"Feature Match: {len(set(user_data.keys()) & set(deployment_package['feature_columns']))}")
                
                # Show which system was more confident
                if system_used == "ai":
                    ai_confidence = calculate_ai_confidence(main_probability)
                    st.write(f"AI Confidence Level: {ai_confidence:.1%}")

            # Rest of your existing results display code...
            # [Keep the risk factors breakdown and recommendations sections the same]


def model_performance(deployment_package):
    """Display model performance information"""
    st.header("🤖 Model Performance")
    
    if deployment_package and 'performance_metrics' in deployment_package:
        st.subheader("📊 Model Performance Metrics")
        metrics = deployment_package['performance_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('Accuracy', 0)*100:.1f}%")
        with col2:
            st.metric("Precision", f"{metrics.get('Precision', 0)*100:.1f}%")
        with col3:
            st.metric("Recall", f"{metrics.get('Recall', 0)*100:.1f}%")
        with col4:
            st.metric("F1-Score", f"{metrics.get('F1-Score', 0)*100:.1f}%")
    else:
        st.info("📊 Performance metrics not available in current mode")
    
    st.subheader("🔧 System Information")
    st.write("**Detection System:** Hybrid AI + Rule-Based")
    st.write("**AI Models:** Random Forest (Primary)")
    st.write("**Confidence Threshold:** 60%")
    st.write("**Fallback System:** Rule-Based Assessment")
    st.write("**Total Features:** 53 comprehensive security metrics")
    
    st.subheader("🎯 Confidence Levels")
    st.write("- **High Confidence:** ≥ 80% - AI results used")
    st.write("- **Medium Confidence:** 60-79% - AI results used") 
    st.write("- **Low Confidence:** < 60% - Rule-based system used")

def about_section():
    """About section for the application"""
    st.header("ℹ️ About This Application")

    st.markdown("""
    ## Mobile Threat Detection System for Elderly Users

    This AI-powered application helps detect and mitigate mobile-specific threats for elderly users
    by analyzing 53 comprehensive security features across multiple dimensions.

    ### 🎯 Key Features:
    - **53-Point Security Assessment**: Comprehensive feature analysis
    - **Real-time Threat Detection**: Analyzes user behavior and device patterns
    - **AI + Rule-Based System**: Combines machine learning with expert rules
    - **Smart Confidence Handling**: Uses AI when confident, rules when uncertain
    - **Risk Assessment**: Comprehensive risk scoring with actionable insights
    - **Security Recommendations**: Personalized security advice

    ### 🔧 Technical Stack:
    - **Machine Learning**: Scikit-learn, Random Forest
    - **Web Framework**: Streamlit
    - **Visualization**: Plotly
    - **Deployment**: Streamlit Cloud

    ### 📊 Feature Categories:
    1. **User Profile & Demographics** (6 features)
    2. **Security Settings & Configuration** (7 features) 
    3. **Usage Patterns & Behavior** (11 features)
    4. **Immediate Risk Factors** (8 features)
    5. **Advanced Security Metrics** (21 features)

    ### 🧠 How Confidence Works:
    - **AI Probability**: How likely this is a threat (0-1)
    - **AI Confidence**: How sure the AI is about its prediction (distance from 0.5)
    - **High Confidence (≥80%)**: AI results are reliable
    - **Low Confidence (<60%)**: Rule-based system takes over
    """)

    st.success("This system is designed specifically for elderly users to enhance their mobile security!")

if __name__ == "__main__":
    main()
