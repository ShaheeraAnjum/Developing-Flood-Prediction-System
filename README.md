# Developing-Flood-Prediction-System
Flood Prediction System is a machine learning project that predicts flood risk using historical weather and environmental data such as rainfall, temperature, humidity, and river levels. It helps support early warning systems, improve disaster preparedness, and enable timely decisions to reduce the impact of floods.
# Rising Waters: Flood Prediction System

## Project Overview
**Rising Waters** is a machine learning-powered flood prediction system designed to address the growing risks of severe floods caused by unpredictable weather patterns, climate change, and rapid urbanization. By leveraging advanced data analysis and predictive algorithms on historical meteorological data, the system identifies potential flood risks before they occur, providing crucial early warnings to government agencies, disaster management authorities, and local communities.

## Objectives
- **Early Warning System:** Detect potential flood risks hours or days in advance using real-world weather parameters.
- **Data-Driven Insights:** Analyze historical rainfall patterns, temperature, cloud visibility, and humidity to predict disaster probability.
- **Enhanced Preparedness:** Enable disaster management teams to proactively plan evacuations and allocate resources efficiently.
- **Robust Modeling:** Compare various machine learning algorithms to select the most accurate and reliable predictive model.

## Features
- **Comprehensive Data Pipeline:** Handles missing values, performs outlier capping using the IQR method, and applies label encoding and standard scaling to raw data.
- **Model Comparison:** Automatically trains and compares Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and XGBoost algorithms.
- **XGBoost Integration:** Deploys a highly accurate XGBoost classifier as the final predictive model.
- **Premium Web Interface:** A Flask-based web application featuring a modern, responsive, dark-mode design with glassmorphism aesthetics.
- **Database Architecture:** Built-in SQLAlchemy schemas to manage Users, Weather Data, Machine Learning Models, and Prediction records efficiently.
- **Interactive Jupyter Notebooks:** Exploratory Data Analysis (EDA) capabilities for univariate and multivariate analysis using Matplotlib and Seaborn.

## Technologies Used
**Backend & Machine Learning:**
- Python 3
- Flask (Web Framework) & Flask-SQLAlchemy (ORM)
- Pandas & NumPy (Data Manipulation)
- Scikit-learn (Preprocessing & Modeling)
- XGBoost (Advanced Classification Algorithm)
- Joblib (Model Serialization)

**Frontend:**
- HTML5
- CSS3 (Vanilla CSS, Glassmorphism, Micro-animations)
- JavaScript

**Data Analysis & Visualization:**
- Jupyter Notebook
- Matplotlib & Seaborn

## Project Structure
```text
Rising Waters/
├── data/
│   └── flood_dataset.csv          # The historical weather dataset
├── model/
│   ├── floods.save                # Trained XGBoost model file
│   └── transform.save             # Fitted StandardScaler file
├── notebooks/
│   └── Data_Analysis.ipynb        # Jupyter notebook for Epic 1 & 2 (EDA)
├── static/
│   ├── main.css                   # Custom stylesheets for the web app
│   └── main.js                    # Frontend interactivity
├── templates/
│   ├── home.html                  # Landing page
│   ├── index.html                 # Prediction input form
│   ├── chance.html                # High flood risk alert page
│   └── no_chance.html             # Safe/No flood risk page
├── app.py                         # Main Flask application and routes
├── train.py                       # ML pipeline for preprocessing and training
├── generate_dataset.py            # Script to generate synthetic dataset (if needed)
├── models.py                      # SQLAlchemy ER Diagram definitions
└── requirements.txt               # Project dependencies
```

## Use Cases
1. **Early Flood Warning and Evacuation Planning:** 
   A meteorologist enters current rainfall (e.g., 1200mm) and cloud visibility readings for a flood-prone district into the web application. The XGBoost model instantly analyzes the inputs and predicts a high probability of flooding, redirecting them to an alert page and allowing authorities to issue evacuation advisories immediately.
   
2. **Disaster Response and Resource Allocation:** 
   A disaster relief coordinator uses the application during monsoon season to monitor multiple regions simultaneously. By entering regional weather data for each area, the system provides instant flood risk classifications, helping prioritize exactly where to deploy emergency rescue teams and resources.
   
3. **Model Validation and Performance Assessment:** 
   A government data analyst tests the models against historical flood event data. By reviewing the outputs from `train.py`, they can confirm that the XGBoost model achieves over 95% accuracy on test data, proving the system’s reliability for operational, real-world deployment.
