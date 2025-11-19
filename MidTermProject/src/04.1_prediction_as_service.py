import joblib 
import pandas as pd
from flask import Flask, request, jsonify

app = Flask('Medical Insurance Cost Prediction')

# UTILS
preprocessor, model = joblib.load("../data/models/model.joblib")

@app.route("/predict", methods=['POST'])
def predict():
    payload = request.get_json(force=True) 
   
    # Preprocess the input request accepted by model
    print("\nDataframe")
    payload_df = pd.DataFrame(payload, index=[0])
    print(payload_df)
    print("\nProcessed data")
    proc_req = preprocessor.transform(payload_df)
    print(proc_req)
    print("\nPrediction")
    print(model.predict(proc_req))
    
    return jsonify({'predictions': model.predict(proc_req).tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=False)