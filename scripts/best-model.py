import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("data/clean_data.csv")

categorical_cols = ['season', 'mnth','weekday','weathersit']
categorical_cols_no_onehot = ['holiday', 'workingday', 'yr']
numerical_cols = ['temp',  'hum', 'windspeed', 'hr_sin', 'hr_cos']

feature_cols = [
    "season", "yr", "mnth", "holiday", "weekday",
    "workingday", "weathersit", "temp", "hum",
    "windspeed", "hr_sin", "hr_cos"
]

X = df[feature_cols]
y = df["cnt"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

cat_pipe_no_onehot = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent'))
])

num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy = 'median'))
])

non_linear_preprocessing = ColumnTransformer([
    ('cat', cat_pipe, categorical_cols),
    ('cat_no_onehot', cat_pipe_no_onehot, categorical_cols_no_onehot),
    ('num', num_pipe, numerical_cols)   
],remainder = 'drop'
)

rf_pipe = Pipeline([
    ('prep', non_linear_preprocessing),
    ('model', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
))
])



rf_pipe.fit(X_train, y_train)

joblib.dump(rf_pipe, "models/bike_model.joblib")
joblib.dump(feature_cols, "models/feature_cols.joblib")