# Air Quality Index (AQI) Prediction System  

**A complete ML pipeline using:**  
- **R** for model training & prediction  
- **Flask** for API serving  
- **Streamlit** for interactive frontend  

---

## 🛠️ **How It Works**  

### **1. Data Flow Architecture**  
```
Streamlit UI (Python) → Flask API (Python) → R Model → Returns Prediction
```  

1. **User Input** collected via Streamlit web interface  
2. **Flask API** receives the request and calls the R prediction script  
3. **R Model** processes input, runs prediction, returns AQI value  
4. **Results** displayed in Streamlit with visualizations  

---

## 📂 **Project Structure**  
```
AQI-Prediction-System/  
├── R/                           # R Modeling Code  
│   ├── train_model.R            # Trains XGBoost model  
│   ├── predict_aqi.R            # Makes predictions  
│   └── models/                  # Saved models  
│       ├── xgb_model.rds        # Trained model  
│       └── preprocessor.rds     # Feature engineering  
├── api/                         # Flask Backend  
│   ├── app.py                   # API endpoints  
│   └── requirements.txt         # Python dependencies  
├── frontend/                    # Streamlit UI  
│   └── app.py                   # User interface  
└── data/                        # Sample datasets  
```

---

## ⚙️ **Setup & Installation**  

### **Prerequisites**  
- Python 3.8+  
- R 4.2+  

### **1. Clone Repository**  
```bash
git clone https://github.com/yourusername/aqi-prediction-system.git  
cd aqi-prediction-system  
```

### **2. Install Python Dependencies**  
```bash
python -m venv venv  
source venv/bin/activate  # Windows: venv\Scripts\activate  
pip install -r api/requirements.txt  
```

### **3. Install R Packages**  
Run in R console:  
```R
install.packages(c("xgboost", "caret", "jsonlite"))  
```

---

## 🚀 **Running the System**  

### **1. Start Flask API**  
```bash
cd api  
flask run --port=5000  
```

### **2. Launch Streamlit Frontend**  
```bash
cd frontend  
streamlit run app.py  
```
➡️ Access UI at: `http://localhost:8501`  

---

## 🔍 **Key Components Explained**  

### **A. R Modeling (`train_model.R`)**  
- **Data Processing**: Handles missing values, creates lag features  
- **Model**: XGBoost regressor trained on historical AQI data  
- **Saves**: Model + preprocessor as `.rds` files  

### **B. Flask API (`app.py`)**  
**Endpoint**:  
```python
@app.route('/predict', methods=['POST'])  
def predict():  
    # Calls R script via subprocess  
    # Returns JSON: {"aqi": 158.2}  
```  
**Test API**:  
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"PM2.5":45, "PM10":80, "NO2":30}'  
```

### **C. Streamlit UI (`app.py`)**  
- **Input Form**: Sliders for pollutant values  
- **Output**: Real-time AQI prediction + health advisory  
- **Visualization**: Plotly charts for historical trends  

---

## 🐛 **Troubleshooting**  

| Issue | Solution |  
|-------|----------|  
| R.dll not found | Set `R_HOME` in environment variables |  
| Package errors | Reinstall dependencies with `install.packages()` |  
| API connection failed | Verify Flask is running on correct port |  

---

## 📜 **License**  
MIT License - Free for academic and commercial use  

**Author**: Your Name  
**Contact**: your.email@example.com  

```diff
+ System ready for deployment! Follow the steps to run locally or deploy to cloud.
```
