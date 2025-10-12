import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def fill_missings(df, numerical_cols, categorical_cols):
    df_filled = df.copy()

    if len(numerical_cols) > 0:
        df_filled[numerical_cols] = df_filled[numerical_cols].fillna(0.0)

    if len(categorical_cols) > 0:
        df_filled[categorical_cols] = df_filled[categorical_cols].fillna("NA")

    return df_filled

def describe_data(data):
    print(data.columns)
    print(data.shape)
    print(data.isnull().sum())

    print("\nNumerical columns summary:")
    print(data.select_dtypes(include=[np.number]).describe().T)
    print("\nCategorical columns summary:")
    data.select_dtypes(include=['object']).apply(lambda ser: print("\n", ser.value_counts()))

    print("\nTarget variable distribution:")
    print(data['converted'].value_counts())
    print(data['converted'].value_counts(normalize=True))

    print("\nFirst few rows:")
    print(data.head())

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=X_test.shape[0], random_state=42)
    print(X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test

data_path = "data/input/course_lead_scoring.csv"
data = pd.read_csv(data_path)
numerical_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=['object']).columns
describe_data(data)

data_filled = fill_missings(data, numerical_cols, categorical_cols)
describe_data(data_filled)


# CORRELATION
print("\nCorrelation matrix for numerical factors:")
correlation_matrix = data_filled[numerical_cols].corr()
print(correlation_matrix)

print("\nFinding maximum correlation pair:")
np.fill_diagonal(correlation_matrix.values, np.nan)
max_corr = correlation_matrix.max().max()
print(f"Maximum correlation value: {max_corr:.4f}")

# Find the pair with maximum correlation
max_corr_pair = correlation_matrix.stack().idxmax()
print(max_corr_pair)
print(correlation_matrix.stack().sort_values())

indices = [
    ('annual_income', 'interaction_count'),
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count')
]
correlation_matrix.stack().loc[indices].idxmax()


# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
y = data_filled['converted']
X = data_filled.drop('converted', axis=1)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# MUTUAL INFORMATION
print("\nCalculating mutual information with 'converted' target:")

# Encode categorical variables for mutual information calculation
X_encoded = X_train.copy()
label_encoders = {}
for col in categorical_cols:
    if col != 'converted':  # Skip target variable
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

# Calculate mutual information
mi_scores = mutual_info_classif(X_encoded.values, y_train.values, random_state=42)
mi_df = pd.Series(np.round(mi_scores,3), index = X_encoded.columns).sort_values()
print(mi_df)
print(mi_df[categorical_cols].max(), mi_df[categorical_cols].idxmax())


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dv = DictVectorizer(sparse=False)

X_train_ohe = dv.fit_transform(X_train.to_dict(orient='records'))
X_test_ohe = dv.transform(X_test.to_dict(orient='records'))
X_val_ohe = dv.transform(X_val.to_dict(orient='records'))

model = Pipeline([
    #('dv', DictVectorizer(sparse=False)),
    #('scaler', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42))
])
model.fit(X_train_ohe, y_train)
print(np.round(model['model'].intercept_,4))
print(np.round(model['model'].coef_,4))

model.score(X_train_ohe, y_train) # return accuracy
print(f"{model.score(X_val_ohe, y_val): .4f}") # return accuracy
all_feature_accuracy = model.score(X_val_ohe, y_val)

acc_list = []
for feature in list(X_train.columns):
    print(feature)
    X_train_ohe = dv.fit_transform(X_train.drop(feature, axis=1).to_dict(orient='records'))
    X_val_ohe = dv.transform(X_val.drop(feature, axis=1).to_dict(orient='records'))
    model = Pipeline([
        ('model', LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42))
    ])
    model.fit(X_train_ohe, y_train)
    print(f"{model.score(X_train_ohe, y_train): .4f}")  # return accuracy
    print(f"{model.score(X_val_ohe, y_val): .4f}") # return accuracy
    acc_list.append(model.score(X_val_ohe, y_val))

accuracies = pd.Series(acc_list, index=list(X_train.columns))

accuracies_diff = accuracies - all_feature_accuracy
print(accuracies_diff.sort_values())


acc_list = []
for C in [0.01, 0.1, 1, 10, 100]:
    print(C)
    X_train_ohe = dv.fit_transform(X_train.to_dict(orient='records'))
    X_val_ohe = dv.transform(X_val.to_dict(orient='records'))
    model = Pipeline([
        ('model', LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42))
    ])
    model.fit(X_train_ohe, y_train)
    print(f"{model.score(X_train_ohe, y_train): .4f}")  # return accuracy
    print(f"{model.score(X_val_ohe, y_val): .4f}") # return accuracy
    acc_list.append(model.score(X_val_ohe, y_val))

accuracies = pd.Series(acc_list, index=[0.01, 0.1, 1, 10, 100])
print(accuracies.sort_values())