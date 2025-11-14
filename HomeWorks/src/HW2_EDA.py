
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error


def fill_missings(train, val, test, mean=0):
    train.loc[:,'horsepower'] = train['horsepower'].fillna(mean)
    test.loc[:,'horsepower'] = test['horsepower'].fillna(mean)
    val.loc[:,'horsepower'] = val['horsepower'].fillna(mean)
    return train, val, test

def train_and_evaluate_linear_regression(model, train, val, test):

    X = train.drop(columns=['fuel_efficiency_mpg']).values
    y = train.fuel_efficiency_mpg
    X_val = val.drop(columns=['fuel_efficiency_mpg']).values
    y_val = val.fuel_efficiency_mpg

    model.fit( X, train['fuel_efficiency_mpg'].values)
    print(model.coef_, model.intercept_)

    y_val_pred = model.predict(X_val)

    #print(y_val.describe(), y_val_pred.describe())

    return root_mean_squared_error(y_val, y_val_pred)

def get_train_val_test_set(data, seed):
    data_sampled = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(data_sampled.shape)
    train = data_sampled.iloc[:int(data_sampled.shape[0]*.6)]
    val = data_sampled.iloc[int(data_sampled.shape[0]*.6):int(data_sampled.shape[0]*.8)]
    test = data_sampled.iloc[int(data_sampled.shape[0]*.8):]
    print(train.shape, val.shape, test.shape)
    return train, val, test

data_path = "data/input/car_fuel_efficiency.csv"
data = pd.read_csv(data_path)
print(data.columns)
print(data.describe())

data = data[[
    'engine_displacement',
    'horsepower',
    'vehicle_weight',
    'model_year',
    'fuel_efficiency_mpg'
]]


print(data.isnull().sum())
print(data.describe())


train, val, test = get_train_val_test_set(data, seed=42)

train_fill0, val_fill0, test_fill0 = fill_missings(train.copy(), val.copy(), test.copy())
print(train.describe())
print(train_fill0.describe())
print(val_fill0.describe())
model = LinearRegression()
print(train_and_evaluate_linear_regression(model, train_fill0, val_fill0, test_fill0)) #.51

train_fillmean, val_fillmean, test_fillmean = fill_missings(train.copy(), val.copy(), test.copy(), mean=train['horsepower'].mean())
print(train.describe())
print(train_fillmean.describe())
model = LinearRegression()
train_and_evaluate_linear_regression(model, train_fillmean, val_fillmean, test_fillmean) #0.461


r = [0, 0.01, 0.1, 1, 5, 10, 100]
rmse_list = []
for r in r:
    print(r)
    model = Ridge(alpha=r)
    rmse = train_and_evaluate_linear_regression(model, train_fill0.copy(), val_fill0.copy(), test_fill0.copy())
    print(rmse)
    rmse_list.append(rmse)

min(rmse_list)


train, val, test = get_train_val_test_set(data, seed=9)
model = Ridge(alpha=0.01)
rmse = train_and_evaluate_linear_regression(
    model,
    pd.concat([train_fill0, val_fill0],axis=0),
    test_fill0.copy(), pd.DataFrame()
)
print(rmse)

rmse_list = []
for seed in range(0, 10):
    print(seed)
    train, val, test = get_train_val_test_set(data, seed=seed)
    train_fill0, val_fill0, test_fill0 = fill_missings(train.copy(), val.copy(), test.copy())
    model = Ridge(alpha=0.01)
    rmse = train_and_evaluate_linear_regression(
        model,
        pd.concat([train_fill0, val_fill0],axis=0),
        test_fill0.copy(), pd.DataFrame()
    )
    rmse_list.append(rmse)
    print(rmse)
rmse_list
import numpy as np
np.std(rmse_list)

train, val, test = get_train_val_test_set(data, seed=9)
train_fill0, val_fill0, test_fill0 = fill_missings(train.copy(), val.copy(), test.copy())
model = Ridge(alpha=0.01)
rmse = train_and_evaluate_linear_regression(
    model,
    pd.concat([train_fill0, val_fill0],axis=0),
    test_fill0.copy(), pd.DataFrame()
)
print(rmse)
