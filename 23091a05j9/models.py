from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'user'
    user_id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    weather_data = db.relationship('WeatherData', backref='user', lazy=True)

class WeatherData(db.Model):
    __tablename__ = 'weather_data'
    data_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    annual_rainfall = db.Column(db.Float, nullable=False)
    cloud_visibility = db.Column(db.Float, nullable=False)
    seasonal_rainfall = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    prediction = db.relationship('Prediction', backref='weather_data', uselist=False)

class MachineLearningModel(db.Model):
    __tablename__ = 'machine_learning_model'
    model_id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    algorithm_type = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    model_file = db.Column(db.String(200), nullable=False)
    predictions = db.relationship('Prediction', backref='ml_model', lazy=True)

class Prediction(db.Model):
    __tablename__ = 'prediction'
    prediction_id = db.Column(db.Integer, primary_key=True)
    data_id = db.Column(db.Integer, db.ForeignKey('weather_data.data_id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('machine_learning_model.model_id'), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    flood_probability = db.Column(db.Float, nullable=True)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
