from flask import Flask, request, jsonify
import rpy2.robjects as robjects
from flask_cors import CORS
import logging
import rpy2.robjects.numpy2ri as numpy2ri  # Change to numpy2ri
from rpy2.robjects.conversion import localconverter
import os
import numpy as np
import threading
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

# Create Flask app 
app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit to call this API

# Activate numpy converter instead of pandas2ri
numpy2ri.activate()

# Initialize R environment
r = robjects.r

# Set R_HOME if needed (especially on Windows)
if os.name == 'nt':  # Windows
    r_home = r'C:\Program Files\R\R-4.4.0'  # Adjust to your R installation
    if os.path.exists(r_home):
        os.environ['R_HOME'] = r_home
        logger.info(f"Set R_HOME to {r_home}")
    else:
        logger.warning(f"R_HOME path {r_home} does not exist. Please update it.")

# Define a thread-local storage for R-related context
thread_local = threading.local()

# Create a decorator to initialize thread-local R context
def ensure_r_context(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Initialize R context for this thread if needed
        if not hasattr(thread_local, 'r_initialized'):
            logger.info(f"Initializing R context for thread {threading.get_ident()}")
            thread_local.r_initialized = True
        return f(*args, **kwargs)
    return decorated

# Load R packages and the model (only once when server starts)
logger.info("Initializing R environment and loading model...")
r('''
library(xgboost)
library(caret)  # Add caret library which is needed for preProcess

# Function to load the model
load_model <- function() {
  tryCatch({
    model <- readRDS("aqi_model_xgb.rds")
    preprocessor <- readRDS("aqi_model_preprocessor.rds")
    return(list(model = model, preprocessor = preprocessor))
  }, error = function(e) {
    print(paste("Error loading model:", e$message))
    return(NULL)
  })
}

# Load model once at startup
model_obj <- load_model()

# Helper function to calculate AQI from pollutant concentrations
calculate_aqi_subindex <- function(value, breakpoints) {
  if (is.na(value)) return(NA)  # Handle NA values
  
  for (bp in breakpoints) {
    low <- bp[1]; high <- bp[2]; sub_low <- bp[3]; sub_high <- bp[4]
    if (!is.na(value) && value >= low && value <= high) {
      return(((sub_high - sub_low) / (high - low)) * (value - low) + sub_low)
    }
  }
  return(NA)
}

# Define breakpoints for AQI calculation
aqi_breakpoints <- list(
  "PM2.5" = list(c(0, 30, 0, 50), c(31, 60, 51, 100), c(61, 90, 101, 200),
                 c(91, 120, 201, 300), c(121, 250, 301, 400), c(251, 500, 401, 500)),
  "PM10" = list(c(0, 50, 0, 50), c(51, 100, 51, 100), c(101, 250, 101, 200),
                c(251, 350, 201, 300), c(351, 430, 301, 400), c(431, 600, 401, 500)),
  "NO2" = list(c(0, 40, 0, 50), c(41, 80, 51, 100), c(81, 180, 101, 200),
               c(181, 280, 201, 300), c(281, 400, 301, 400), c(401, 1000, 401, 500))
)

# Calculate AQI for a specific pollutant
calculate_pollutant_aqi <- function(pollutant, value) {
  if (pollutant == "PM2.5") {
    return(calculate_aqi_subindex(value, aqi_breakpoints$`PM2.5`))
  } else if (pollutant == "PM10") {
    return(calculate_aqi_subindex(value, aqi_breakpoints$PM10))
  } else if (pollutant == "NO2") {
    return(calculate_aqi_subindex(value, aqi_breakpoints$NO2))
  } else {
    return(NA)
  }
}

# Function to preprocess data
preprocess_data <- function(data) {
  # Use predict method from caret package
  processed <- predict(model_obj$preprocessor, data)
  return(processed)
}

# Function to predict AQI
predict_aqi <- function(pm25_aqi, pm10_aqi, no2_aqi, aqi_lag1, aqi_lag2, aqi_lag3) {
  # Create data frame with input features
  new_data <- data.frame(
    PM2.5_AQI = pm25_aqi,
    PM10_AQI = pm10_aqi,
    NO2_AQI = no2_aqi,
    AQI_Lag1 = aqi_lag1,
    AQI_Lag2 = aqi_lag2,
    AQI_Lag3 = aqi_lag3
  )
  
  # Custom preprocessing using our wrapper function
  new_data_scaled <- preprocess_data(new_data)
  
  # Make predictions
  predictions <- predict(model_obj$model, as.matrix(new_data_scaled))
  
  return(predictions)
}
''')
logger.info("R environment initialized successfully")

# Endpoint to get AQI for a pollutant
@app.route('/calculate_aqi', methods=['POST'])
@ensure_r_context
def calculate_aqi():
    try:
        data = request.get_json()
        pollutant = data.get('pollutant')
        value = data.get('value')
        # Get city if available (optional)
        city = data.get('city', 'Unknown location')
        
        if not pollutant or value is None:
            return jsonify({"error": "Missing required parameters"}), 400
        
        logger.info(f"Calculating AQI for pollutant: {pollutant}, value: {value}, city: {city}")
        
        # Convert any NumPy types to Python native types
        if isinstance(value, (np.integer, np.floating)):
            value = float(value)
            
        # Use robjects directly to create R objects
        with localconverter(robjects.default_converter + numpy2ri.converter):
            # Explicitly create R character and numeric values
            r_pollutant = robjects.StrVector([str(pollutant)])
            r_value = robjects.FloatVector([float(value)])
            
            # Call the R function
            r_calc_aqi = robjects.globalenv['calculate_pollutant_aqi']
            result = r_calc_aqi(r_pollutant[0], r_value[0])
            
            # Check if the result is NA
            is_na = robjects.r('is.na')(result)[0]
            if is_na:
                return jsonify({"aqi": None, "message": "Cannot calculate AQI for given value"})
            
            # Convert result to Python float
            aqi_value = float(result[0])
            return jsonify({"aqi": aqi_value, "city": city})
    except Exception as e:
        logger.error(f"Error calculating AQI: {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint to predict AQI
@app.route('/predict', methods=['POST'])
@ensure_r_context
def predict():
    try:
        data = request.get_json()
        
        # Get required parameters
        pm25_aqi = data.get('pm25_aqi')
        pm10_aqi = data.get('pm10_aqi')
        no2_aqi = data.get('no2_aqi')
        aqi_lag1 = data.get('aqi_lag1')
        aqi_lag2 = data.get('aqi_lag2')
        aqi_lag3 = data.get('aqi_lag3')
        # Get city if available (optional)
        city = data.get('city', 'Unknown location')
        
        # Validate inputs
        if None in [pm25_aqi, pm10_aqi, no2_aqi, aqi_lag1, aqi_lag2, aqi_lag3]:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Convert any NumPy types to Python native types
        pm25_aqi = float(pm25_aqi)
        pm10_aqi = float(pm10_aqi)
        no2_aqi = float(no2_aqi)
        aqi_lag1 = float(aqi_lag1)
        aqi_lag2 = float(aqi_lag2)
        aqi_lag3 = float(aqi_lag3)
        
        logger.info(f"Predicting AQI with parameters: PM2.5={pm25_aqi}, PM10={pm10_aqi}, NO2={no2_aqi}, " 
                   f"Lag1={aqi_lag1}, Lag2={aqi_lag2}, Lag3={aqi_lag3}, City={city}")
        
        # Use localconverter with numpy2ri
        with localconverter(robjects.default_converter + numpy2ri.converter):
            # Create R vectors
            r_pm25_aqi = robjects.FloatVector([pm25_aqi])
            r_pm10_aqi = robjects.FloatVector([pm10_aqi])
            r_no2_aqi = robjects.FloatVector([no2_aqi])
            r_aqi_lag1 = robjects.FloatVector([aqi_lag1])
            r_aqi_lag2 = robjects.FloatVector([aqi_lag2])
            r_aqi_lag3 = robjects.FloatVector([aqi_lag3])
            
            # Call the R function
            r_predict = robjects.globalenv['predict_aqi']
            result = r_predict(
                r_pm25_aqi[0], r_pm10_aqi[0], r_no2_aqi[0], 
                r_aqi_lag1[0], r_aqi_lag2[0], r_aqi_lag3[0]
            )
            
            # Convert result to Python float
            predicted_aqi = float(result[0])
            return jsonify({"predicted_aqi": predicted_aqi, "city": city})
    except Exception as e:
        logger.error(f"Error predicting AQI: {e}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

# Home endpoint with basic instructions
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "AQI Prediction API",
        "endpoints": {
            "/health": "Health check endpoint",
            "/calculate_aqi": "POST endpoint to calculate AQI for a pollutant",
            "/predict": "POST endpoint to predict AQI based on multiple inputs"
        } 
    })

if __name__ == '__main__':
    # Set threaded=False to avoid rpy2 threading issues
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=False)