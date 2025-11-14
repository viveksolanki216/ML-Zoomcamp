import requests

if __name__ == "__main__":

    pred_request = {
            "chronic_count": 6, 
            "is_high_risk": 1, 
            "days_hospitalized_last_3yrs": 20,
            "age": 60,
            "smoker": "Never",
            "bmi": 50, 
            "hospitalizations_last_3yrs": 2, 
            "visits_last_year": 5
    }
    url = "http://127.0.0.1:9696/predict"

    predictions = requests.post(url, json=pred_request).json()
    print(predictions)