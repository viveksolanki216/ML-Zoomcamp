import joblib 
import pandas as pd

# UTILS
def load_model(model_path):
    preprocessor, model = joblib.load(model_path)
    return preprocessor, model

def predict(request, model, preprocessor):
    # Preprocess the input request accepted by model
    print("\nProcessed data")
    proc_req = preprocessor.transform(pd.DataFrame(request, index=[0]))
    print(proc_req)
    print("\nPrediction")
    print(model.predict(proc_req))
    return model.predict(proc_req)

if __name__ == "__main__":

    request = {
            'chronic_count': 6, 
            'is_high_risk': 1, 
            'days_hospitalized_last_3yrs': 20,
            'age': 60,
            'smoker': 'Never',
            'bmi': 50, 
            'hospitalizations_last_3yrs': 2, 
            'visits_last_year': 5
    }
    print(request)
    
    # LOAD MODELS
    model_path = "../data/models/model.joblib"
    preprocessor, model = load_model(model_path)

    # PREDICT
    prediction = predict(request, model, preprocessor)