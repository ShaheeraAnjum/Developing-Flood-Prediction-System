import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

warnings.filterwarnings('ignore')

def preprocess_data(df):
    # 1. Handle Missing Values
    print("Handling missing values...")
    if df.isnull().any().any():
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                
    # 2. Outlier Handling (Capping using IQR)
    print("Capping outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('class')
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, np.where(df[col] < lower_bound, lower_bound, df[col]))
        
    # 3. Handle Categorical Values (Label Encoding)
    print("Label encoding categorical variables...")
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        # In a real app, you might want to save the label encoders as well.
        
    return df, label_encoders

def decisiontree(X_train, X_test, y_train, y_test):
    print("\n--- Decision Tree ---")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model, acc

def randomForest(X_train, X_test, y_train, y_test):
    print("\n--- Random Forest ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return model, acc

def KNN(X_train, X_test, y_train, y_test):
    print("\n--- K-Nearest Neighbors ---")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return model, acc

def xgboost_model(X_train, X_test, y_train, y_test):
    print("\n--- XGBoost ---")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return model, acc

def compareModel(accuracies):
    print("\n=== Model Comparison ===")
    for name, acc in accuracies.items():
        print(f"{name}: {acc*100:.2f}%")
        
    best_model_name = max(accuracies, key=accuracies.get)
    print(f"\nBest Model Selected: {best_model_name}")

def main():
    # Load dataset
    print("Loading data...")
    if not os.path.exists('data/flood_dataset.csv'):
        print("Dataset not found. Please run generate_dataset.py first.")
        return
        
    df = pd.read_csv('data/flood_dataset.csv')
    
    # Preprocess
    df, _ = preprocess_data(df)
    
    # Split Data
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling
    print("Applying StandardScaler...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Save the scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(sc, 'model/transform.save')
    print("Scaler saved to model/transform.save")
    
    # Train Models
    dt_mod, dt_acc = decisiontree(X_train, X_test, y_train, y_test)
    rf_mod, rf_acc = randomForest(X_train, X_test, y_train, y_test)
    knn_mod, knn_acc = KNN(X_train, X_test, y_train, y_test)
    xgb_mod, xgb_acc = xgboost_model(X_train, X_test, y_train, y_test)
    
    accuracies = {
        'Decision Tree': dt_acc,
        'Random Forest': rf_acc,
        'KNN': knn_acc,
        'XGBoost': xgb_acc
    }
    
    compareModel(accuracies)
    
    # Since XGBoost is defined as the best deployment model in the requirements
    print("\nSaving final deployment model (XGBoost)...")
    joblib.dump(xgb_mod, 'model/floods.save')
    print("Model saved to model/floods.save")

if __name__ == "__main__":
    main()
