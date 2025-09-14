# 🐄 Cattle Milk Yielding and Health Prediction System

A full-stack web application designed to help farmers and agricultural professionals monitor cattle health and predict milk yield. The system uses a machine learning backend built with Python and scikit-learn to analyze various cattle attributes and provides a user-friendly dashboard for data visualization and interaction.

---

## 🚀 Features
- **Milk Yield Prediction**: Predicts expected milk yield based on breed, age, weight, and lactation period.  
- **Health Status Prediction**: Classifies cattle health as *Excellent, Good, Fair,* or *Poor* based on vitals and historical data.  
- **Interactive Dashboard**: Displays real-time predictions and key metrics.  
- **Data Visualization**: Charts for milk yield distribution and health status breakdown.  
- **RESTful API**: Flask-based API for handling requests and predictions.  
- **Local Data Storage**: SQLite database for managing cattle records.  

---

## 🛠️ Technologies Used

### Backend
- **Python** – Core backend language  
- **Flask** – RESTful API framework  
- **scikit-learn** – ML models (RandomForestRegressor & RandomForestClassifier)  
- **Pandas & NumPy** – Data manipulation and numerical operations  
- **Matplotlib & Seaborn** – Data visualizations  
- **SQLite3** – Lightweight local database  

### Frontend
- **HTML5, CSS3, JavaScript** – Web dashboard  
- **Chart.js** – Interactive charts  
- **Font Awesome** – UI icons  
- **Fetch API** – Asynchronous API requests  

---

## 📂 Project Structure
```
cattle-monitoring-system/
│── Cattle_monitoring_system_final.py   # Flask backend with ML models & database
│── frontend.html                       # Complete single-file frontend dashboard
│── README.md                           # Project documentation
```

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.6+ installed

### Backend Setup
```bash
# Clone repository
git clone https://github.com/your-username/cattle-monitoring-system.git
cd cattle-monitoring-system

# Create virtual environment
python -m venv venv
# Activate it
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install Flask flask-cors pandas scikit-learn matplotlib seaborn

# Run the backend server
python Cattle_monitoring_system_final.py
```
Server runs at: **http://127.0.0.1:5000**

### Frontend Setup
- Open `frontend.html` in a web browser.  
- Ensure backend server is running for API connectivity.  

---

## 🔗 API Endpoints

### **POST /api/predict**
Predicts milk yield or health status.  
**Request Body:**
```json
{ "type": "yield", "data": { ... } }
{ "type": "health", "data": { ... } }
```
**Response:**
```json
{ "prediction": value, "unit": "liters" }
{ "prediction": "Status" }
```

### **GET /api/animals**
Returns a list of stored animal data.  
**Response:**
```json
[ { "id": "1", "milk_yield": 14.5 }, ... ]
```

### **GET /api/charts/yield-distribution**
Returns milk yield distribution chart (base64).  
**Response:**
```json
{ "chart": "base64_string" }
```

### **GET /api/charts/health-breakdown**
Returns health status breakdown chart (base64).  
**Response:**
```json
{ "chart": "base64_string" }
```

---

## 📌 Future Enhancements
- IoT integration for real-time cattle monitoring  
- Mobile app support  
- Cloud deployment for scalability  
- Multilingual farmer-friendly interface  

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
