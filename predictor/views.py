import os
import joblib
import pandas as pd
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE_DIR, "predictor/ml/car_price_rf.pkl")
)

preprocessor = joblib.load(
    os.path.join(BASE_DIR, "predictor/ml/preprocessor.pkl")
)

def index(request):
    prediction = None
    submitted_data = None

    if request.method == "POST":
        mileage = request.POST.get("mileage")
        engine_size = request.POST.get("engine_size")
        car_age = request.POST.get("car_age")
        fuel_type = request.POST.get("fuel_type")
        manufacturer = request.POST.get("manufacturer")

        submitted_data = {
            "mileage": mileage,
            "engine_size": engine_size,
            "car_age": car_age,
            "fuel_type": fuel_type,
            "manufacturer": manufacturer,
        }

        df = pd.DataFrame({
            "mileage": [float(mileage)],
            "engine_size": [float(engine_size)],
            "car_age": [int(car_age)],
            "fuel_type": [fuel_type],
            "manufacturer": [manufacturer],
        })
        
        GBP_TO_INR = 105

        X = preprocessor.transform(df)
        prediction = round(model.predict(X)[0] * GBP_TO_INR)
        prediction = f"{prediction:,}"

    # IMPORTANT: we DO NOT send form values back → form resets
    return render(
        request,
        "predictor/index.html",
        {
            "prediction": prediction,
            "submitted_data": submitted_data
        }
    )

























# GBP_TO_INR = 105