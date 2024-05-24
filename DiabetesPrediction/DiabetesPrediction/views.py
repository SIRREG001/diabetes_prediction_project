from django.shortcuts import render
from joblib import load
from django.http import HttpResponse

# Import the libraries installed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
# import libraries for prediction
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the trained model
# trained_model = load('trained_model.joblib')

# Create your views here.


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    diabetes_data = pd.read_csv(
        r"C:/Users/UdochukwuReginald/Desktop/diabetes_prediction_project/diabetes.csv")
    # Train Test Split
    X = diabetes_data.drop("Outcome", axis=1)
    y = diabetes_data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    # Train the Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    Pregnancies = float(request.GET['Pregnancies'])
    Glucose = float(request.GET['Glucose'])
    BloodPressure = float(request.GET['BloodPressure'])
    SkinThickness = float(request.GET['SkinThickness'])
    Insulin = float(request.GET['Insulin'])
    BMI = float(request.GET['BMI'])
    DiabetesPedigreeFunction = float(
        request.GET['DiabetesPedigreeFunction'])
    Age = float(request.GET['Age'])
    input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                   Insulin, BMI, DiabetesPedigreeFunction, Age]]
    y_pred = model.predict(input_data)
    result = ""
    if y_pred == [1]:
        result = "Postive, You have diabetes"
    else:
        result = "Negative, You do not have diabetes"
    return render(request, 'predict.html', {"result": result})
