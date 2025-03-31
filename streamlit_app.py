import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import time
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌬️",
    layout="wide"
)

# Define API endpoint
API_URL = "http://localhost:5000"  # Change this if your Flask app is on a different host/port

# App title and description
st.title("🌬️ Air Quality Index Prediction")
st.markdown("""
This dashboard allows you to calculate AQI values for individual pollutants and predict overall AQI based on multiple parameters.
""")

# Define list of cities
cities = [
    "Kurla", "Bandra", "CSMT", "Colaba", "BKC", "Dadar", "Andheri"
    
]

# Check if Flask API is running
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return False

# Add retry mechanism
@st.cache_data(ttl=10)  # Cache for 10 seconds
def api_health_with_retry(max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        if check_api_health():
            return True
        else:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                time.sleep(retry_delay)
    return False

# Check API health with retry
if not api_health_with_retry():
    st.error("⚠️ The Flask API service is not running. Please start the API service to use this application.")
    st.stop()
else:
    st.success("✅ Connected to AQI prediction service")
    
# Create tabs
tab1, tab2, tab3 = st.tabs(["Calculate AQI", "Predict AQI", "About AQI"])

# Tab 1: Calculate AQI for individual pollutants
with tab1:
    st.header("Calculate AQI for Individual Pollutants")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Add city selection to Tab 1
        city = st.selectbox(
            "Select City",
            cities,
            key="city_tab1",
            help="Choose the city for which you want to calculate the AQI"
        )
        
        pollutant = st.selectbox(
            "Select Pollutant",
            ["PM2.5", "PM10", "NO2"],
            help="Choose the pollutant for which you want to calculate the AQI"
        )
        
        value = st.number_input(
            f"Enter {pollutant} concentration (μg/m³)",
            min_value=0.0,
            value=50.0,
            step=0.1,
            help=f"Input the concentration of {pollutant} in μg/m³"
        )
        
        calculate_button = st.button("Calculate AQI")
        
        if calculate_button:
            with st.spinner("Calculating AQI..."):
                try:
                    # Make API request with proper error handling
                    response = requests.post(
                        f"{API_URL}/calculate_aqi",
                        json={"pollutant": pollutant, "value": value, "city": city},
                        timeout=10  # Set a reasonable timeout
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        aqi_value = data.get("aqi")
                        
                        if aqi_value is not None:
                            st.success(f"The AQI for {pollutant} concentration of {value} μg/m³ in {city} is: **{aqi_value:.1f}**")
                            
                            # Determine AQI category
                            if aqi_value <= 50:
                                category = "Good"
                                color = "green"
                            elif aqi_value <= 100:
                                category = "Satisfactory"
                                color = "lightgreen"
                            elif aqi_value <= 200:
                                category = "Moderate"
                                color = "yellow"
                            elif aqi_value <= 300:
                                category = "Poor"
                                color = "orange"
                            elif aqi_value <= 400:
                                category = "Very Poor"
                                color = "red"
                            else:
                                category = "Severe"
                                color = "darkred"
                            
                            st.markdown(f"<div style='background-color:{color}; padding:10px; border-radius:5px'><h3 style='text-align:center; color:white'>AQI Category: {category}</h3></div>", unsafe_allow_html=True)
                        else:
                            st.warning(f"The AQI could not be calculated for {pollutant} with value {value} in {city}. {data.get('message', '')}")
                    else:
                        error_message = "Unknown error"
                        try:
                            error_data = response.json()
                            error_message = error_data.get('error', 'Unknown error')
                        except:
                            pass
                        st.error(f"Error from API (Status {response.status_code}): {error_message}")
                        
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The server might be busy or unavailable.")
                except requests.exceptions.ConnectionError:
                    st.error("Connection error. Please check if the API server is running.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
    
    with col2:
        st.subheader("AQI Reference Chart")
        aqi_chart_data = pd.DataFrame({
            "Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
            "AQI Range": ["0-50", "51-100", "101-200", "201-300", "301-400", "401-500"],
            "Color": ["green", "lightgreen", "yellow", "orange", "red", "darkred"]
        })
        
        # Create a custom table with colored cells
        st.markdown(
            """
            <style>
            .aqi-table {
                width: 100%;
                border-collapse: collapse;
                text-align: center;
            }
            .aqi-table th, .aqi-table td {
                padding: 8px;
                border: 1px solid #ddd;
            }
            .aqi-table th {
                background-color: #f2f2f2;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
        
        html_table = "<table class='aqi-table'><tr><th>Category</th><th>AQI Range</th></tr>"
        for _, row in aqi_chart_data.iterrows():
            html_table += f"<tr><td style='background-color:{row['Color']}; color:white'>{row['Category']}</td><td>{row['AQI Range']}</td></tr>"
        html_table += "</table>"
        
        st.markdown(html_table, unsafe_allow_html=True)
        
        st.subheader("Health Implications")
        health_data = pd.DataFrame({
            "Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
            "Health Impact": [
                "Minimal impact", 
                "Minor breathing discomfort to sensitive people", 
                "Breathing discomfort to people with lung disease, children and older adults",
                "Breathing discomfort to most people on prolonged exposure",
                "Respiratory illness on prolonged exposure",
                "Affects healthy people and seriously impacts those with existing diseases"
            ]
        })
        
        st.dataframe(health_data, hide_index=True)

# Tab 2: Predict AQI using multiple parameters
with tab2:
    st.header("Predict Future AQI")
    
    st.info("Enter the individual pollutant AQI values and the AQI values from previous days to predict future AQI.")
    
    # City selection at the top of Tab 2
    city_pred = st.selectbox(
        "Select City",
        cities,
        key="city_tab2",
        help="Choose the city for which you want to predict the AQI"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        pm25_aqi = st.number_input("PM2.5 AQI", min_value=0.0, value=80.0, step=1.0)
        pm10_aqi = st.number_input("PM10 AQI", min_value=0.0, value=70.0, step=1.0)
        no2_aqi = st.number_input("NO2 AQI", min_value=0.0, value=60.0, step=1.0)
    
    with col2:
        aqi_lag1 = st.number_input("Previous day AQI (1 day ago)", min_value=0.0, value=90.0, step=1.0)
        aqi_lag2 = st.number_input("Previous day AQI (2 days ago)", min_value=0.0, value=85.0, step=1.0)
        aqi_lag3 = st.number_input("Previous day AQI (3 days ago)", min_value=0.0, value=95.0, step=1.0)
    
    predict_button = st.button("Predict AQI")
    
    if predict_button:
        with st.spinner(f"Predicting AQI for {city_pred}..."):
            try:
                # Make API request
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "city": city_pred,
                        "pm25_aqi": pm25_aqi,
                        "pm10_aqi": pm10_aqi,
                        "no2_aqi": no2_aqi,
                        "aqi_lag1": aqi_lag1,
                        "aqi_lag2": aqi_lag2,
                        "aqi_lag3": aqi_lag3
                    },
                    timeout=10  # Set a reasonable timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predicted_aqi = data.get("predicted_aqi")
                    
                    # Create columns for the results display
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.subheader(f"Prediction Result for {city_pred}")
                        st.success(f"Predicted AQI: **{predicted_aqi:.1f}**")
                        
                        # Determine AQI category
                        if predicted_aqi <= 50:
                            category = "Good"
                            color = "green"
                        elif predicted_aqi <= 100:
                            category = "Satisfactory"
                            color = "lightgreen"
                        elif predicted_aqi <= 200:
                            category = "Moderate"
                            color = "yellow"
                        elif predicted_aqi <= 300:
                            category = "Poor"
                            color = "orange"
                        elif predicted_aqi <= 400:
                            category = "Very Poor"
                            color = "red"
                        else:
                            category = "Severe"
                            color = "darkred"
                        
                        st.markdown(f"<div style='background-color:{color}; padding:10px; border-radius:5px'><h3 style='text-align:center; color:white'>Predicted Category: {category}</h3></div>", unsafe_allow_html=True)
                    
                    with result_col2:
                        # Create historical + prediction chart
                        today = datetime.now()
                        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3, 0, -1)]
                        dates.append(today.strftime("%Y-%m-%d"))
                        dates.append((today + timedelta(days=1)).strftime("%Y-%m-%d"))
                        
                        historical_values = [aqi_lag3, aqi_lag2, aqi_lag1]
                        current_and_pred = [np.mean([pm25_aqi, pm10_aqi, no2_aqi]), predicted_aqi]
                        
                        aqi_values = historical_values + current_and_pred
                        categories = ["Historical"] * 3 + ["Current"] + ["Predicted"]
                        
                        # Create chart data
                        chart_data = pd.DataFrame({
                            "Date": dates,
                            "AQI": aqi_values,
                            "Type": categories
                        })
                        
                        try:
                            fig = px.line(
                                chart_data, 
                                x="Date", 
                                y="AQI",
                                color="Type",
                                title=f"AQI Trend and Prediction for {city_pred}",
                                markers=True
                            )
                            
                            # Add horizontal lines for AQI categories
                            fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
                            fig.add_hline(y=100, line_dash="dash", line_color="lightgreen", annotation_text="Satisfactory")
                            fig.add_hline(y=200, line_dash="dash", line_color="yellow", annotation_text="Moderate")
                            fig.add_hline(y=300, line_dash="dash", line_color="orange", annotation_text="Poor")
                            fig.add_hline(y=400, line_dash="dash", line_color="red", annotation_text="Very Poor")
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as chart_error:
                            st.error(f"Error generating chart: {str(chart_error)}")
                else:
                    error_message = "Unknown error"
                    try:
                        error_data = response.json()
                        error_message = error_data.get('error', 'Unknown error')
                    except:
                        pass
                    st.error(f"Error from API (Status {response.status_code}): {error_message}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The server might be busy or unavailable.")
            except requests.exceptions.ConnectionError:
                st.error("Connection error. Please check if the API server is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# Tab 3: About AQI
with tab3:
    st.header("About Air Quality Index (AQI)")
    
    st.markdown("""
    ### What is AQI?
    
    The Air Quality Index (AQI) is a scale used to communicate how polluted the air currently is or how polluted it is forecast to become. The AQI focuses on health effects you may experience within a few hours or days after breathing polluted air.
    
    ### How is AQI calculated?
    
    The AQI is calculated for major air pollutants regulated by the Clean Air Act: ground-level ozone, particle pollution (PM2.5 and PM10), carbon monoxide (CO), sulfur dioxide (SO2), and nitrogen dioxide (NO2).
    
    For each of these pollutants, EPA has established national air quality standards to protect public health. The higher the AQI value, the greater the level of air pollution and the greater the health concern.
    
    ### AQI Categories
    
    | AQI Range | Category | Health Implications |
    |-----------|----------|---------------------|
    | 0-50 | Good | Air quality is considered satisfactory, and air pollution poses little or no risk. |
    | 51-100 | Satisfactory | Air quality is acceptable; however, for some pollutants, there may be a moderate health concern for a very small number of people. |
    | 101-200 | Moderate | Members of sensitive groups may experience health effects. The general public is not likely to be affected. |
    | 201-300 | Poor | Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects. |
    | 301-400 | Very Poor | Health warnings of emergency conditions. The entire population is more likely to be affected. |
    | 401-500 | Severe | Health alert: everyone may experience more serious health effects. |
    
    ### AQI and Health
    
    When AQI values are above 100, air quality is considered to be unhealthy, at first for certain sensitive groups of people, then for everyone as AQI values increase.
    
    - **Sensitive Groups**: People with lung disease, older adults, and children are at greater risk from exposure to air pollution.
    - **Health Effects**: Common air pollutants can cause a range of health problems, from burning eyes and irritated respiratory system to more serious effects such as damage to lung tissue and cancer.
    
    ### References
    
    - [EPA AQI Basics](https://www.airnow.gov/aqi/aqi-basics/)
    - [WHO Air Quality Guidelines](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health)
    """)

# Footer
st.markdown("---")
st.markdown("📊 AQI Prediction Dashboard | Built with Streamlit & Flask")