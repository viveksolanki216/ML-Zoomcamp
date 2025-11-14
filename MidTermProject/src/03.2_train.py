import pandas as pd
import numpy as np
from utils import get_config_params
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


# UTILS
def get_error(y_test, y_test_pred, y_train, y_train_pred):
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print("Train RMSE:",train_rmse, "-- Test RMSE:", test_rmse)
    return test_rmse, train_rmse

if __name__ == "__main__":
    # GET CONFIG PARAMS
    input_data_path = get_config_params('00_config.yaml')
    print(input_data_path)

    # LOAD DATA
    data = pd.read_csv(input_data_path)
    print(data.shape)
    data.head()

    # DEFINE TARGET AND I/P FACTORS
    target='annual_medical_cost'
    factors = [ 
            'chronic_count' , 'is_high_risk', 'days_hospitalized_last_3yrs', 'age',
            'smoker', 'bmi', 'hospitalizations_last_3yrs', 'visits_last_year'
            ]
    data = data[[target] + factors].reset_index(drop=True)
    print(data.shape)

    # REMOVE OUTLIERS
    data = data[data['annual_medical_cost'] <= 5000].reset_index(drop=True)

    print(data['annual_medical_cost'].describe())
    print(pd.Series(np.log2(data['annual_medical_cost']).describe()))

    # TRAIN TEST SPLIT: 80% Train 20% Test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")

    X_train = train_data.drop(columns=['annual_medical_cost'])
    y_train = train_data['annual_medical_cost']
    X_test = test_data.drop(columns=['annual_medical_cost'])
    y_test = test_data['annual_medical_cost']


    # PREPROCESS DATA
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    print(categorical_cols)

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', ohe, categorical_cols)
        ]
    )
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print(X_train.shape)
    # Since the above transformet generates a numpy matrix and not a pandas dataframe
    # Manually generate feature names
    feature_names = (
        numeric_cols + 
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    )
    print(f"Total features after encoding: {len(feature_names)}")

    # TRAINING MODEL
    # Training a Linear Regression Model with l2 penalty - Best Parameters
    model_ridge = Ridge(alpha=10)
    model_ridge.fit(X_train, y_train)
    y_train_pred = model_ridge.predict(X_train)
    y_test_pred = model_ridge.predict(X_test)
    errors = get_error(y_test, y_test_pred, y_train, y_train_pred)
    #print(model_ridge.coef_)
    #print(y_train_pred[0:10])
    #print(y_test_pred[0:10])

    #STORING MODEL AND DATA PREPREOCESSOR
    models_dir = "../data/models"
    joblib.dump((preprocessor, model_ridge), f"{models_dir}/model.joblib")
