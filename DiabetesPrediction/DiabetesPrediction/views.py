from django.shortcuts import render
from joblib import load
from django.http import HttpResponse

# Load the trained model
# trained_model = load('trained_model.joblib')

# Create your views here.


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    return render(request, 'predict.html')
