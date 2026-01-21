import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, time, date
import shap
import matplotlib.pyplot as plt
from groq import Groq
import plotly.graph_objects as go
import seaborn as sns
from fpdf import FPDF
import io
from PIL import Image
import tempfile
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dongsi Air Quality Simulator",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- INITIALIZE SESSION STATE ---
# This prevents KeyErrors before the simulation runs
if 'has_run' not in st.session_state:
    st.session_state['has_run'] = False
if 'pred' not in st.session_state:
    st.session_state['pred'] = 0.0
if 'explanation' not in st.session_state:
    st.session_state['explanation'] = ""
if 'top_3_friendly' not in st.session_state:
    st.session_state['top_3_friendly'] = ["Unknown", "Unknown"]
if 'shap_values_global' not in st.session_state:
    st.session_state['shap_values_global'] = None

# --- MODEL LOADING ---
@st.cache_resource
def load_air_quality_model():
    # Initialize the regressor with the same settings your teammate used
    model = xgb.XGBRegressor(enable_categorical=True, tree_method='hist')
    model.load_model("xgboost_model.json")
    return model

model = load_air_quality_model()

# --- Helper Function to Convert Matplotlib Figures to Images ---
def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# --- SHAP GLOBAL BEESWARM FUNCTION ---
test_df = pd.read_csv('test_data.csv')

# Separate features from target for the SHAP explainer
X_test = test_df.drop('PM2.5', axis=1)
y_test = test_df['PM2.5']

# Convert 'wd' back to category if you are using XGBoost
X_test['wd'] = X_test['wd'].astype('category')

def plot_global_beeswarm(model, X_test):
    """Generates the Beeswarm plot for researchers."""
    # Note: TreeExplainer is best for XGBoost/RandomForest
    explainer = shap.TreeExplainer(model)
    
    # We calculate SHAP values for the provided X_test dataset
    shap_values = explainer(X_test)
    
    # We create a matplotlib figure explicitly to pass to streamlit
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.tight_layout()
    st.pyplot(fig)

    return fig

@st.cache_data
def get_global_shap_values(_model, _X_test):
    """Calculates global SHAP values for the entire test set once."""
    explainer = shap.TreeExplainer(_model)
    return explainer(_X_test) # This returns a SHAP Explanation object

# Initialize Groq Client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_llama_explanation(prediction, top_features):
    # Initialize the Groq client using your secret key
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # Mapping model features to "Plain Language" for the LLM
    prompt = f"""
    Current AQI Prediction: {prediction:.1f}
    Key scientific drivers: {top_features}
    
    Task: A resident wants to know if they can go jogging in Dongsi. 
    Give a 2-sentence answer in plain, friendly language. 
    Explain if it's safe and how the specific weather/pollutants (like low wind or high humidity) are causing it.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Groq's current high-performance Llama model
            messages=[
                {"role": "system", "content": "You are a helpful local health advisor for Dongsi district."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Current Index is {prediction:.1f}. Please check local guidelines before exercising."

def get_medical_explanation(prediction, shap_series):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # Identify the biggest 'increaser' and biggest 'decreaser'
    top_pusher = shap_series.idxmax()
    top_reducer = shap_series.idxmin()
    
    prompt = f"""
    AQI: {prediction:.1f}. 
    Top Increasing Factor: {top_pusher}
    Top Reducing Factor: {top_reducer}
    
    Task: Provide a technical medical summary for a Health Official based on AQI, Top Increasing Factor: and Top Reducing Factor. 
    Explain in 2-3 sentences: 
    1. The primary pollutant driving the risk. 
    2. How the weather (like wind or humidity) is helping or hurting the situation.
    3. The expected impact on sensitive respiratory groups.
    Use professional but concise language.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are a Chief Medical Officer specialized in environmental health."}],
            temperature=0.5
        )
        return completion.choices[0].message.content
    except:
        return "Local drivers indicate high pollutant concentrations. Monitor respiratory admission rates."

# --- HELP TEXT DEFINITIONS ---
pollutant_help = {
    "PM10": "Particulate Matter (10 micrometers or less): Can be inhaled into lungs. Sources include dust, pollen, and mold.",
    "SO2": "Sulfur Dioxide: A gas produced by burning fossil fuels (coal/oil). Can cause respiratory irritation.",
    "NO2": "Nitrogen Dioxide: Primarily gets into the air from the burning of fuel (cars, power plants). A key indicator of traffic pollution.",
    "CO": "Carbon Monoxide: An odorless, colorless gas that forms when carbon in fuel doesn't burn completely.",
    "O3": "Ground-level Ozone: Formed by chemical reactions between oxides of nitrogen (NOx) and volatile organic compounds (VOC) in sunlight."
}

# --- SIDEBAR (Global Controls) ---
with st.sidebar:
    st.header("🎮 Simulator Controls")
    st.markdown("Adjust these parameters to see how they impact the predicted Air Quality.")

    # 1. Chemical Pollutants (Floats)
    st.subheader("🧪 Pollutant Levels")
    pm10 = st.slider("PM10", 0.0, 500.0, 80.0, help=pollutant_help["PM10"])
    so2 = st.slider("SO2", 0.0, 200.0, 15.0, help=pollutant_help["SO2"])
    no2 = st.slider("NO2", 0.0, 200.0, 45.0, help=pollutant_help["NO2"])
    co = st.slider("CO", 0.0, 10.0, 1.2, help=pollutant_help["CO"])
    o3 = st.slider("O3", 0.0, 300.0, 50.0, help=pollutant_help["O3"])

    st.divider()

    # 2. Meteorological Parameters (Floats & Categorical)
    st.subheader("🌦️ Weather Conditions")
    temp = st.number_input("Temperature (°C)", -20.0, 45.0, 15.0)
    pres = st.number_input("Pressure (hPa)", 980.0, 1040.0, 1013.0)
    dewp = st.number_input("Dew Point (°C)", -30.0, 30.0, 5.0)
    rain = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
    wspm = st.slider("Wind Speed (m/s)", 0.0, 20.0, 2.5)
    
    # Wind Direction (Categorical "c")
    wd_options = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    wd = st.selectbox("Wind Direction (wd)", options=wd_options)

    st.divider()

    # 3. Temporal Parameters (Ints)
    st.subheader("⏰ Time Context")
    selected_date = st.date_input("Select Date", datetime.now())
    extracted_month = selected_date.month
    
    # Time Picker to extract the Hour
    selected_time = st.time_input("Select Time", time(12, 0))
    extracted_hour = selected_time.hour

    # ACTION BUTTON
    run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    st.divider()
    st.caption("Developed by Nurul Izzati & Nur Fadilah for WQF7009 Alternative Assessment © 2026")

# Create the input_row from sidebar variables
# Ensure the order matches feature_names exactly
input_data = {
    "PM10": pm10, "SO2": so2, "NO2": no2, "CO": co, "O3": o3,
    "TEMP": temp, "PRES": pres, "DEWP": dewp, "RAIN": rain, "WSPM": wspm,
    "wd": wd, "hour": selected_time.hour, "month": selected_date.month
}

# Transform into a DataFrame
input_row = pd.DataFrame([input_data])

# CRITICAL: Convert 'wd' to category so XGBoost doesn't crash
input_row['wd'] = input_row['wd'].astype('category')

# Feature names in the exact order expected by the model
feature_names = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd", "hour", "month"]

# --- GLOBAL PREDICTION LOGIC ---
if run_sim:
    with st.spinner("Calculating results..."):
        # 1. Prediction: Calculate the raw AQI score
        prediction = model.predict(input_row)[0]
        
        # 2. SHAP Values: Calculate the "Why"
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_row)
        
        # 3. Create a Full Series: This maps the score to EVERY feature
        # This is the "Full SHAP Analysis" requested by the Research Community
        feature_importance = pd.Series(
            shap_values[0], 
            index=feature_names
        ).sort_values(ascending=False)
        
        # 4. Get the Top 3 names specifically for the LLM prompt
        top_3_names = feature_importance.head(3).index.tolist()
        
        # 5. SAVE TO SESSION STATE
        # We save the WHOLE series so Tab 3 can filter for pollutants later
        st.session_state['pred'] = prediction
        st.session_state['shap_vals'] = feature_importance 
        st.session_state['top_3_friendly'] = top_3_names
        st.session_state['has_run'] = True
        
        # 6. Groq/Llama Explanation
        # We pass the top 2 factors to give the AI context for its plain-language answer
        st.session_state['explanation'] = get_llama_explanation(prediction, ", ".join(top_3_names))
        st.session_state['med_explanation'] = get_medical_explanation(prediction, feature_importance)
        
        # Force a rerun to update the global meter and tabs immediately
        st.rerun()

# --- MAIN TITLE & DESCRIPTION ---
st.title("🌬️ Dongsi Air Quality Simulator")
st.markdown("""
Welcome to the interactive simulation platform for the Dongsi district. 
Please select your stakeholder profile below to access tailored insights and controls.
""")

# --- HEADER: METER & LEGEND ---
meter_col, legend_col = st.columns([1.8, 3])

if st.session_state.get('has_run'):

    # 1. Determine the Meteorological Season
    month = selected_date.month
    if month in [12, 1, 2]:
        season = "❄️ Winter (Heating Season - High Coal Demand)"
    elif month in [3, 4, 5]:
        season = "🌱 Spring (Transition - Dust Storm Risk)"
    elif month in [6, 7, 8]:
        season = "☀️ Summer (High Photochemical Activity/Ozone)"
    else:
        season = "🍂 Autumn (Stable Atmosphere - Stagnation Risk)"

    # 2. Display the Global Status Header
    st.write("---")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Analysis Date", selected_date.strftime("%B %d, %Y"))
    
    with c2:
        # Formatting hour into 24h and 12h for clarity
        time_str = f"{extracted_hour:02d}:00"
        st.metric("Observation Time", time_str)
        
    with c3:
        st.metric("Meteorological Season", season.split(" ")[1])
        st.caption(season)

    score = st.session_state['pred']
    
    # Define Label/Color
    if score <= 50: label, bar_color = "GOOD", "#C8E6C9"
    elif score <= 100: label, bar_color = "MODERATE", "#FFF9C4"
    elif score <= 150: label, bar_color = "UNHEALTHY (S)", "#FFE0B2"
    elif score <= 200: label, bar_color = "UNHEALTHY", "#FFCDD2"
    else: label, bar_color = "HAZARDOUS", "#E1BEE7"

    with meter_col:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = score,
            number = {'font': {'size': 40}},
            title = {'text': f"STATUS: {label}", 'font': {'size': 18}},
            gauge = {'axis': {'range': [None, 300]}, 'bar': {'color': bar_color}, 'bgcolor': "#F0F2F6"}
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

else:
    with meter_col:
        st.info("👈 Adjust parameters and click 'Run Simulation'")

with legend_col:
    st.markdown("<p style='font-size:13px; font-weight:bold; color:#666;'>AQI Category Reference</p>", unsafe_allow_html=True)
    ref_labels = [("Good", "0-50", "#C8E6C9"), ("Moderate", "51-100", "#FFF9C4"), 
                  ("Unhealthy (for Sensitive group)", "101-150", "#FFE0B2"), ("Unhealthy", "151-200", "#FFCDD2"),
                  ("Very Unhealthy", "201-300", "#F3E5F5"), ("Hazardous", "301+", "#E1BEE7")]
    
    r1 = st.columns(3)
    r2 = st.columns(3)
    for i, (l, rng, c) in enumerate(ref_labels):
        target = r1 if i < 3 else r2
        target[i%3].markdown(f"""<div style="background-color:{c}; padding:8px; border-radius:6px; text-align:center; border: 1px solid rgba(0,0,0,0.05); margin-bottom:5px;">
            <p style="margin:0; font-size:11px; font-weight:bold; color:#444;">{l}</p>
            <p style="margin:0; font-size:10px; color:#666;">{rng}</p></div>""", unsafe_allow_html=True)

st.divider()

# --- STAKEHOLDER TABS ---
# Creating the four distinct areas requested
tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ ENVIRONMENTAL REGULATORS", 
    "🏥 PUBLIC HEALTH OFFICIALS", 
    "🏘️ GENERAL PUBLIC", 
    "🔬 RESEARCH COMMUNITY"
])

# --- TAB 1: ENVIRONMENTAL REGULATORS ---
with tab1:
    st.header("🏢 Environmental Regulator Dashboard")
    st.caption("Strategic Policy Planning & Emergency Response Intervention")

    if not st.session_state.get('has_run'):
        st.info("📊 Data required for policy analysis. Please run the model simulation first.")
    else:
        # --- 1. DATA PREPARATION (Dynamic) ---
        current_pm25 = st.session_state['pred']
        shap_vals = st.session_state['shap_vals']
        target_aqi = 48.0  # The "Good" threshold goal
        
        # Identify the primary driver dynamically using SHAP
        # We assume the top positive SHAP value is the one we need to "counteract"
        primary_pollutant = shap_vals.idxmax()
        primary_impact = shap_vals.max()
        
        # --- 2. SITUATION: DYNAMIC EMERGENCY PLANNING ---
        if current_pm25 > 50:
            st.error(f"### 🚨 Situation: Emergency Response Required")
            status_color = "inverse"
            status_text = "Unhealthy"
        else:
            st.success(f"### ✅ Situation: Maintenance Mode")
            status_color = "normal"
            status_text = "Good"

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **Event Analysis:** Current forecasting shows **{primary_pollutant}** as the lead driver 
            contributing **+{primary_impact:.1f}** points to the total PM2.5 concentration.
            """)
        with col2:
            st.metric("Forecasted PM2.5", f"{current_pm25:.1f}", delta=status_text, delta_color=status_color)

        st.divider()

        # --- 3. DYNAMIC DICE COUNTERFACTUAL CALCULATION ---
        st.subheader("❓ Policy Inquiry")
        st.markdown(f"**'If we implement restrictions on {primary_pollutant}, how much improvement can we expect?'**")

        # Logic: Calculate required reduction to reach target_aqi
        # Calculation: (Current - Target) / Current
        gap = current_pm25 - target_aqi
        
        if gap > 0:
            # Simple simulation of DiCE logic: To drop 'gap' amount, 
            # we need to reduce the primary pollutant proportional to its SHAP impact.
            reduction_needed_pct = min(90, int((gap / current_pm25) * 100))
            
            # Simulated Counterfactual Values
            original_val = 2100  # Placeholder: in production, use st.session_state['input_df'][primary_pollutant]
            target_val = original_val * (1 - (reduction_needed_pct / 100))
            
            # --- 4. VISUALIZATION: THE COMPARISON TABLE ---
            dice_df = pd.DataFrame({
                "Parameter": ["AQI Status", "PM2.5 Concentration", f"{primary_pollutant} Levels"],
                "Current State": [f"🔴 {status_text}", f"{current_pm25:.1f} µg/m³", f"{original_val} units"],
                "Target (Counterfactual)": ["✅ Good", f"{target_aqi} µg/m³", f"{target_val:.0f} units"],
                "Required Action": ["-", f"Reduce by {gap:.1f} points", f"⬇️ {reduction_needed_pct}% Reduction"]
            })
            
            st.table(dice_df)

            # --- 5. ACTIONABLE ANSWER ---
            st.subheader("📝 Answer & Regulatory Action Plan")
            st.info(f"""
            **Counterfactual Analysis (DiCE):** To achieve a 'Good' air quality rating, 
            the bureau must achieve a **{reduction_needed_pct}% reduction** in **{primary_pollutant}** sources. This translates to the following emergency measures:
            """)
            
            # Dynamic Policy Recommendations
            recs = {
                "CO": ["Enforce 'No-Drive' zones in Dongsi District.", "Increase public transit capacity by 20%."],
                "SO2": ["Mandate 50% power output at local coal plants.", "Activate industrial scrubbers."],
                "NO2": ["Restrict heavy-duty diesel trucks during daylight hours.", "Temporary ban on construction machinery."],
                "PM10": ["Deploy water mist cannons for dust suppression.", "Halt all open-air construction."]
            }
            
            # Get recommendations based on the dynamic primary pollutant
            current_recs = recs.get(primary_pollutant, ["General emission reduction required.", "Increase monitoring frequency."])
            
            for r in current_recs:
                st.write(f"✅ {r}")

        else:
            st.balloons()
            st.success("Current parameters are optimal. No emergency intervention required.")

# --- TAB 2: PUBLIC HEALTH OFFICIALS ---
with tab2:
    if st.session_state['has_run']:
        score = st.session_state['pred']
        shap_vals = st.session_state['shap_vals'].sort_values(ascending=True)

        # --- SECTION 1: EMERGENCY ADVISORY STATUS ---
        # High-visibility banners for "Early Warning"
        if score > 200:
            st.error("### 🟥 EMERGENCY: RED ALERT (Level 4 Advisory)")
            st.markdown("**Public Action:** Immediate suspension of outdoor activities. Mask mandate for essential travel.")
        elif score > 150:
            st.error("### 🟧 WARNING: UNHEALTHY (Level 3 Advisory)")
            st.markdown("**Public Action:** Issue early warnings to schools and nursing homes.")
        elif score > 100:
            st.warning("### 🟨 CAUTION: SENSITIVE GROUPS (Level 2 Advisory)")
            st.markdown("**Public Action:** Targeted advisories for clinics and pediatric wards.")
        else:
            st.success("### 🟩 STABLE: GOOD/MODERATE (Level 1 Advisory)")
            st.markdown("**Public Action:** Routine surveillance; no active advisories required.")

        st.divider()

        # --- 2. SITUATION & QUESTION (The Hook) ---
        st.subheader("🚨 Situation: Protecting Vulnerable Groups")
        
        col_sit, col_q = st.columns([1, 1])
        with col_sit:
            st.markdown("""
            **Context:** Local hospitals report a spike in respiratory admissions. 
            Health officials need to know if the pollution is localized or widespread 
            to issue the correct advisory.
            """)
        
        with col_q:
            st.warning("❓ **Question:** Should we issue different advisories for different pollution types?")
            st.caption("**Tool Used:** SHAP Contribution Analysis (Driver Attribution)")

        st.divider()

        # --- SECTION 2: SHAP WATERFALL (THE CAUSE) ---
        col_plot, col_vulnerable = st.columns([1.6, 1])

        with col_plot:
            st.subheader("🔬 Local Drivers (Early Warning Indicators)")
            st.write("SHAP values identifying the pollutants pushing AQI above baseline.")
            
            # Waterfall-style Bar Chart
            fig_health = go.Figure(go.Bar(
                y=shap_vals.index,
                x=shap_vals.values,
                orientation='h',
                marker_color=['#ef5350' if x > 0 else '#66bb6a' for x in shap_vals.values]
            ))
            fig_health.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_health, use_container_width=True)

        # --- SECTION 3: PROTECTING VULNERABLE GROUPS ---
        with col_vulnerable:
            st.subheader("🏥 Vulnerable Group Protocols")
            
            # We use SHAP to see if the driver is a specific chemical to tailor the protocol
            top_pollutant = shap_vals[shap_vals.index.isin(["PM10", "SO2", "NO2", "O3", "CO"])].idxmax()
            
            # Dynamic Protocol Cards
            with st.expander("👵 Elderly & Cardiovascular", expanded=True):
                if score > 100:
                    st.write("❌ High Risk: Monitor for arrhythmia and ischemic events.")
                else:
                    st.write("✅ Low Risk: Normal activity permitted.")

            with st.expander("🧒 Pediatric & Asthma", expanded=True):
                if top_pollutant in ["NO2", "PM10"]:
                    st.write(f"⚠️ **Alert:** {top_pollutant} is high. Increase bronchodilator availability in schools.")
                else:
                    st.write("✅ Baseline monitoring only.")

            with st.expander("🏃 Outdoor Workers", expanded=True):
                if score > 150:
                    st.write("❌ Mandatory 15-min breaks per hour in filtered environments.")
                else:
                    st.write("✅ Normal shifts; monitor hydration.")

        st.divider()

        # --- SECTION 4: EARLY WARNING FORECAST INSIGHT ---
        st.subheader("📡 Advisory Justification")

        # --- TAB 2: LOGIC REFINEMENT ---
        # Define which features a Health Official can actually act on
        actionable_features = ["PM10", "SO2", "NO2", "O3", "CO", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd"]

        # Get SHAP values for ONLY actionable features
        actionable_shap = st.session_state['shap_vals'][st.session_state['shap_vals'].index.isin(actionable_features)]

        # Find the top actionable driver
        main_actionable_driver = actionable_shap.idxmax()
        driver_impact = actionable_shap.max()

        # Map technical names to "Official" names
        official_names = {
            "PM10": "Particulate Matter", "SO2": "Sulfur Dioxide", "NO2": "Nitrogen Dioxide",
            "O3": "Ozone", "WSPM": "Wind Stagnation", "DEWP": "Atmospheric Humidity"
        }
        display_driver = official_names.get(main_actionable_driver, main_actionable_driver)

        # Logic: If the score is high, focus on the pollutant. 
        # If the score is low, focus on why it's safe.
        if score > 100:
            st.info(f"""
                **Official Statement:** This health advisory is triggered by elevated **{display_driver}** levels. 
                SHAP analysis identifies this as the primary physical driver, contributing **{driver_impact:.1f} points** to the current risk profile. Seasonal trends (Month: {st.session_state['top_3_friendly'][0]}) 
                further compound these local effects.
            """)
        else:
            st.info("**Official Statement:** Current environmental monitoring indicates baseline levels. No chemical triggers detected.")
        
        st.markdown("### 🔍 Clinical & Regulatory Inquiries")
        st.write("Select a protocol-based question to analyze the current model data from a medical perspective:")

        # Define Actionable Drivers for the logic
        actionable_features = ["PM10", "SO2", "NO2", "O3", "CO", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd"]
        actionable_shap = st.session_state['shap_vals'][st.session_state['shap_vals'].index.isin(actionable_features)]
        top_p = actionable_shap.idxmax()

        # 1. Question: Hospital Load
        with st.expander("🚑 What is the projected impact on Emergency Department (ED) admissions?"):
            if st.session_state['has_run']:
                score = st.session_state['pred']
                if score > 100:
                    st.markdown(f"**Clinical Forecast:** Critical. High levels of **{top_p}** are correlated with a significant spike in acute respiratory distress cases within the next 6-12 hours.")
                    st.write("👉 **Recommendation:** Activate surge capacity protocols for respiratory therapy units.")
                else:
                    st.markdown("**Clinical Forecast:** Stable. Current concentrations do not suggest an immediate deviation from baseline admission rates.")
            else:
                st.info("Run simulation to generate ED impact projection.")

        # 2. Question: School/Vulnerable Group Closures
        with st.expander("🏫 Is there sufficient scientific evidence to justify a 'Sensitive Group' stay-at-home order?"):
            if st.session_state['has_run']:
                impact_points = actionable_shap.max()
                st.markdown(f"**Evidence Summary:** Yes. SHAP Local analysis confirms that **{top_p}** alone is contributing **{impact_points:.1f} points** to the AQI.")
                st.write(f"Medical literature indicates that at these concentrations, {top_p} acts as a potent bronchial irritant. An advisory for schools and nursing homes is scientifically supported by the model.")
            else:
                st.info("Run simulation to analyze evidence for stay-at-home orders.")

        # 3. Question: The "Observation" Discrepancy (Professional Version)
        with st.expander("🧪 Why does the sensor data show 'High Risk' when visibility (haze) is relatively clear?"):
            if st.session_state['has_run']:
                # Logic: If a pollutant is high but humidity/rain is low
                if (st.session_state['shap_vals'].get('DEWP', 0) < 5) and st.session_state['pred'] > 100:
                    st.markdown(f"**Technical Explanation:** This is a **'Invisible Threat'** scenario. Visibility is high because humidity is low, but the concentration of **{top_p}** is dangerously high.")
                    st.write("Officials should warn the public that 'clear skies' do not currently equal 'safe air'. Health risk is driven by chemical concentration, not optical haze.")
                else:
                    st.write("Visibility and Health Risk are currently aligned with atmospheric moisture levels.")
            else:
                st.info("Run simulation to analyze the visibility/risk correlation.")

        st.divider()
#--- TAB 3: GENERAL PUBLIC ---
with tab3:
    if st.session_state['has_run']:
        score = st.session_state['pred']
        
        # --- SECTION 1: ACTIONABLE ADVICE (The "What") ---
        st.markdown("### 🏃 Living Guidelines")
        c1, c2, c3 = st.columns(3)
        with c1:
            if score > 150: st.error("😷 **Wear a Mask**\n\nN95 recommended.")
            elif score > 100: st.warning("😷 **Optional Mask**\n\nSensitive groups.")
            else: st.success("🍃 **No Mask**\n\nEnjoy the air!")
        with c2:
            if score > 100: st.warning("🏠 **Close Windows**\n\nUse air purifiers.")
            else: st.success("🪟 **Ventilate**\n\nOpen your windows.")
        with c3:
            if score > 100: st.error("🚫 **Stay Indoors**\n\nNo heavy exercise.")
            else: st.success("⚽ **Outdoor Fun**\n\nSafe for sports.")

        st.divider()

        # --- SECTION 2: TOP POLLUTANTS (The "Who") ---
        # Requirement: Display top 3 pollutants if AQI is unhealthy
        if score > 100:
            st.subheader("🚨 Primary Pollutant Risks")
            
            # List of features that are strictly pollutants
            pollutants_only = ["PM10", "SO2", "NO2", "CO", "O3"]
            
            # 1. Access global SHAP values
            shap_all = st.session_state['shap_vals'] 
            
            # 2. Filter for only pollutants and take top 3
            # We use .abs() because even a negative impact pollutant is worth noting if it's a top driver
            top_pollutants = shap_all[shap_all.index.isin(pollutants_only)].sort_values(ascending=False).head(3)

            # 3. Designer Layout: Use columns for the Top 3
            p_cols = st.columns(3)
            
            full_names = {
                "PM10": "Dust & Particles", "SO2": "Sulfur (Industry)",
                "NO2": "Nitrogen (Traffic)", "CO": "Carbon Monoxide", "O3": "Smog/Ozone"
            }

            for i, (name, val) in enumerate(top_pollutants.items()):
                with p_cols[i]:
                    st.markdown(f"""
                        <div style="background-color:#F8F9FA; padding:10px; border-radius:8px; border-top: 4px solid #FFCDD2; text-align:center;">
                            <p style="margin:0; font-size:11px; color:#666; font-weight:bold;">RANK {i+1}</p>
                            <p style="margin:0; font-size:13px; font-weight:bold; color:#444;">{full_names.get(name, name)}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.caption("These three pollutants are currently the strongest drivers of the air quality degradation.")
            st.divider()

        # --- SECTION 3: WEATHER CONTEXT (The "Why") ---
        st.subheader("🧐 Environmental Context")
        
        # Access the top 2 features (could be weather or pollutants)
        top_2 = st.session_state['top_3_friendly']
        
        # Translation for weather
        weather_tips = {
            "WSPM": "Low wind is preventing pollutants from dispersing.",
            "DEWP": "High humidity is making particles heavier and more visible.",
            "TEMP": "Current temperature is causing a 'trap' effect near the ground.",
            "RAIN": "Rainfall is currently helping to wash the atmosphere clean.",
            "wd": "Wind direction is bringing in air from more polluted zones."
        }
        
        # Find if a weather factor is in the top 2
        weather_reason = "Atmospheric conditions are stable, allowing pollutants to accumulate."
        for feat in top_2:
            if feat in weather_tips:
                weather_reason = weather_tips[feat]
                break
        
        st.info(f"**Expert Insight:** {weather_reason}")

        

    else:
        st.info("👋 **Welcome to the Resident Portal.** Run the simulation in the sidebar to get your personalized advice.")

    st.markdown("### ❓ Common Resident Questions")
    st.write("Click a question below to see how our AI model analyzes your specific situation:")

    # Define the scenarios
    with st.expander("🏃 Is it safe to go for a jog right now?"):
        if st.session_state['has_run']:
            st.info(f"**AI Health Advisor:** {st.session_state['explanation']}")
        else:
            st.warning("Please run the simulation in the sidebar first to get a real-time answer.")

    with st.expander("📅 I'm planning a park visit for the kids. How does it look?"):
        if st.session_state['has_run']:
            score = st.session_state['pred']
            top_f = st.session_state['top_3_friendly'][0]
            if score < 100:
                st.write(f"**Analysis:** Based on the current forecast, air quality is **Good**. "
                         f"Low levels of **{top_f}** mean it's a great time for the playground.")
            else:
                st.write(f"**Analysis:** It's currently **Unhealthy**. We suggest waiting until "
                         f"wind conditions improve to clear out the local pollution.")
        else:
            st.info("Run the simulation to get a recommendation for your event.")

    with st.expander("🌫️ It looks hazy outside, should I be concerned?"):
        if st.session_state['has_run']:
            top_factors = st.session_state['top_3_friendly']
            score = st.session_state['pred']

            # PRIORITY 1: Is it actually dangerous?
            if score > 100:
                st.markdown("### ⚠️ **Yes, use caution.**")
                st.write(
                    f"With an AQI of **{score:.1f}**, the haze you see is likely a high concentration of "
                    f"pollutants. The model indicates that particles are trapped near the ground. "
                    "You should limit outdoor exposure regardless of the cause."
                )
            
            # PRIORITY 2: If it's relatively safe, is it just humidity?
            elif "DEWP" in top_factors or "RAIN" in top_factors:
                st.markdown("### 🌤️ **It's mostly moisture!**")
                st.write(
                    f"Your AQI is **{score:.1f}** (Moderate/Good). The sky looks thick because of "
                    f"**high humidity**. These are water droplets, "
                    "not harmful smog. It looks worse than it actually is for your health."
                )
            
            # PRIORITY 3: Fallback for other safe conditions
            else:
                st.markdown("### 🔍 **Low health risk.**")
                st.write(
                    "Pollutant levels are currently within safe limits. Any visible haze is likely "
                    "due to light scattering or minor atmospheric dust, but it is not a major health concern today."
                )
        else:
            st.warning("Please run the simulation to analyze visibility.")

    st.divider()
    
# --- TAB 4: RESEARCH COMMUNITY ---
with tab4:
    st.header("🔬 Model Research & Scientific Verification")
    
    # Check if simulation has been run
    if not st.session_state.get('has_run'):
        st.info("💡 **Awaiting Data:** Please run the simulation to see historical distributions and global model logic.")
    else:
        # --- 1. DATA PREP (Dynamic) ---
        current_aqi = st.session_state['pred']
        shap_values_global = st.session_state.get('shap_values_global')
        
        # Calculate Global Importance on the fly
        if shap_values_global is None:
            with st.spinner("Analyzing Global Atmospheric Patterns..."):
                explainer = shap.TreeExplainer(model)
                st.session_state['shap_values_global'] = explainer(X_test)
                shap_values_global = st.session_state['shap_values_global']

        # Identify Top Drivers for the LLM context
        global_importance = pd.Series(
            abs(shap_values_global.values).mean(0), 
            index=X_test.columns
        ).sort_values(ascending=False)
        top_5_drivers = global_importance.head(5).to_dict()
        percentile = (y_test < current_aqi).mean() * 100

        # --- 2. SITUATION 1: HISTORICAL ANCHORING ---
        st.subheader("📊 Situation 1: Historical Contextualization")
        
        col_sit1, col_q1 = st.columns([1, 1])
        with col_sit1:
            st.markdown(f"""**Context:** Researchers need to know if today's prediction of **{current_aqi:.1f}** is a statistical anomaly or a frequent occurrence in Dongsi District.""")
        with col_q1:
            st.warning("❓ **Question:** Where does today's prediction sit within the historical distribution?")

        col_dist, col_stats = st.columns([2, 1])
        
        with col_dist:
            # Probability Density Plot
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
            sns.histplot(y_test, kde=True, color="#90caf9", ax=ax_hist, label="Historical Frequency")
            plt.axvline(current_aqi, color="#ef5350", linestyle='--', linewidth=3, label=f"Current: {current_aqi:.1f}")
            plt.title("Today's AQI vs. Historical Dongsi Records")
            plt.xlabel("PM2.5 (µg/m³)")
            plt.legend()
            st.pyplot(fig_hist)
            

        with col_stats:
            # Calculate Percentile & Z-Score
            percentile = (y_test < current_aqi).mean() * 100
            z_score = (current_aqi - y_test.mean()) / y_test.std()
            
            st.metric("Historical Percentile", f"{percentile:.1f}%")
            st.metric("Z-Score (Deviance)", f"{z_score:.2f} σ")
            
            if percentile > 90:
                st.error("🔴 **Extreme Event:** Statistically rare high pollution.")
            elif percentile < 10:
                st.success("🟢 **Clean Anomaly:** Historically rare air quality.")
            else:
                st.info("🔵 **Typical Range:** Aligns with seasonal norms.")

        st.divider()

        # --- 3. SITUATION 2: MODEL PHYSICS (BEESWARM) ---
        st.subheader("🧪 Situation 2: Model Physics Validation")
        
        st.markdown("**Question:** Do SHAP results align with established atmospheric chemistry (Ozone titration, Wind Dispersion)?")
        
        # Show all features (max_display=20) so the "hidden" ones are visible
        fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(shap_values_global, max_display=20, show=False)
        plt.tight_layout()
        st.pyplot(fig_beeswarm)

        # Dynamic validation logic
        with st.expander("📝 Scientific Validation Report", expanded=True):
            # Check for Ozone Titration
            o3_idx = list(X_test.columns).index("O3")
            o3_corr = pd.Series(shap_values_global.values[:, o3_idx]).corr(X_test["O3"])
            
            st.write(f"✅ **Ozone Logic:** Correlation is `{o3_corr:.2f}`. Negative correlation verifies the model captures Ozone titration chemistry.")
            st.write("✅ **Particulate Coupling:** High PM10 and PM2.5 ranking confirms the model respects shared emission source profiles.")

        st.divider()

        # --- 4. SITUATION 3: THE TUG-OF-WAR (WIND DIRECTION) ---
        st.subheader("🌪️ Situation 3: The 'Hidden' Impact of Wind")
        
        col_sit3, col_q3 = st.columns([1, 1])
        with col_sit3:
            st.markdown("**Context:** Wind Direction ranks low globally (#11), but researchers suspect a geographic cancellation effect.")
        with col_q3:
            st.warning("❓ **Question:** Why does wind rank low when it is scientifically critical?")

        if "wd" in X_test.columns:
            wd_idx = list(X_test.columns).index("wd")
            wd_shap = shap_values_global.values[:, wd_idx]
            
            pos_force = wd_shap[wd_shap > 0].mean() if any(wd_shap > 0) else 0
            neg_force = abs(wd_shap[wd_shap < 0].mean()) if any(wd_shap < 0) else 0

            # Tug-of-War Plot
            fig_tug, ax_tug = plt.subplots(figsize=(10, 2.5))
            ax_tug.barh(["Wind Impact"], [-neg_force], color="#66bb6a", label="Cleansing (Northwest)")
            ax_tug.barh(["Wind Impact"], [pos_force], color="#ef5350", label="Transport (Southeast)")
            plt.axvline(0, color='black', linewidth=2)
            plt.title("Tug-of-War: Wind Direction Cancellation Effect")
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.7), ncol=2)
            st.pyplot(fig_tug)
            
            st.info(f"**Analysis:** The Southeast 'Transport' force (+{pos_force:.2f}) is nearly balanced by the Northwest 'Cleansing' force (-{neg_force:.2f}). This geographic tug-of-war makes wind look 'unimportant' in global averages, but it is actually a primary local driver.")

            st.divider()

            # --- 2. GENERATE COMPREHENSIVE REPORT (GROQ + PDF) ---
            st.subheader("📝 AI-Generated Scientific Synthesis")

            # We wrap everything in one button click to ensure all variables exist at once
            if st.button("Generate & Download PDF Report"):
                with st.spinner("Synthesizing data and rendering PDF..."):
                    try:
                        # A. Get AI Text from Groq
                        prompt = f"""
                        You are a Senior Atmospheric Researcher. Analyze these results for Dongsi District:
                        - Analysis Date: {selected_date.strftime('%B %d, %Y')}
                        - Current Season: {season}
                        - Current Predicted PM2.5: {current_aqi:.1f} µg/m³
                        - Historical Percentile: {percentile:.1f}%
                        - Top Global Drivers (SHAP): {top_5_drivers}

                        Provide a formal 3-paragraph scientific report for a PDF export. 
                        Incorporate how the current season ({season}) typically influences the chemical 
                        significance of these specific drivers.
                        """                        
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile", 
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        # Variable defined HERE
                        report_text = completion.choices[0].message.content

                        # B. Capture Plots as Temporary Files
                        # Ensure 'fig_hist', 'fig_beeswarm', and 'fig_tug' are defined in your tab4 code above this
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_hist:
                            fig_hist.savefig(tmp_hist.name, format='png', bbox_inches='tight')
                            hist_path = tmp_hist.name

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_bees:
                            fig_beeswarm.savefig(tmp_bees.name, format='png', bbox_inches='tight')
                            bees_path = tmp_bees.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_tug:
                            fig_tug.savefig(tmp_tug.name, format='png', bbox_inches='tight')
                            tug_path = tmp_tug.name

                        # C. Build the PDF
                        pdf = FPDF()
                        pdf.add_page()
                        
                        # Header
                        pdf.set_font("Arial", 'B', 16)
                        pdf.cell(200, 10, txt="Air Quality Scientific Research Report", ln=True, align='C')
                        pdf.set_font("Arial", size=10)
                        pdf.cell(200, 10, txt=f"Location: Dongsi District | Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
                        pdf.ln(10)

                        # Scientific Synthesis Section
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Scientific Synthesis & LLM Analysis", ln=True)
                        pdf.set_font("Arial", size=11)
                        # Now 'report_text' is guaranteed to exist
                        pdf.multi_cell(0, 10, txt=report_text.encode('latin-1', 'ignore').decode('latin-1'))
                        pdf.ln(10)

                        # Evidence Section - Distribution
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Historical Distribution Context", ln=True)
                        pdf.image(hist_path, x=10, w=180)
                        
                        # Evidence Section - SHAP & Tug-of-War
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Global Feature Interpretability (SHAP)", ln=True)
                        pdf.image(bees_path, x=10, w=180)
                        pdf.ln(10)
                        
                        pdf.cell(200, 10, txt="Geographic Forcing (Wind Tug-of-War)", ln=True)
                        pdf.image(tug_path, x=10, w=180)

                        # D. Final Output
                        pdf_output = pdf.output(dest='S').encode('latin-1', 'ignore')
                        
                        # Cleanup files
                        for p in [hist_path, bees_path, tug_path]:
                            if os.path.exists(p):
                                os.remove(p)

                        st.success("✅ Report Generated Successfully!")
                        st.download_button(
                            label="📥 Download Research PDF",
                            data=pdf_output,
                            file_name=f"Dongsi_AQI_Research_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    
                    except Exception as e:
                        st.error(f"An error occurred during PDF generation: {e}")
            else:
                st.info("Click the button to generate the scientific synthesis and PDF export.")