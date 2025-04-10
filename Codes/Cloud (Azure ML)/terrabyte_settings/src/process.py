import argparse
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    data = data.drop_duplicates()

    data = data.astype({
        'humidite': 'float32', 'temperature': 'float32', 'EC': 'float32', 'pH': 'float32',
        'teneur_n': 'float32', 'teneur_p': 'float32', 'teneur_k': 'float32',
        'last_fertilization': 'int8', 'target_production': 'int32'
    })

    insert_loc = data.columns.get_loc('type_sol')
    data = pd.concat([data.iloc[:, :insert_loc], pd.get_dummies(data[['type_sol']]), data.iloc[:, insert_loc + 1:]], axis=1)

    insert_loc = data.columns.get_loc('etape')
    data = pd.concat([data.iloc[:, :insert_loc], pd.get_dummies(data[['etape']]), data.iloc[:, insert_loc + 1:]], axis=1)

    return data.reset_index(drop=True)

def create_pipeline() -> Pipeline:
    numeric_features = ['humidite', 'temperature', 'EC', 'pH', 'teneur_n', 'teneur_p', 'teneur_k', 'target_production']
    preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_features)], remainder='passthrough')

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    stacked_model = StackingRegressor(
        estimators=[('rf', rf)],
        final_estimator=gbm,
        passthrough=True,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', MultiOutputRegressor(stacked_model))
    ])

    return pipeline

def train_and_eval_model(pipeline: Pipeline, data: pd.DataFrame, registered_model_name: str):
    
    data = preprocess_data(data)

    X = data.drop(columns=["besoins"], errors='ignore')
    y = data["besoins"].apply(eval).apply(pd.Series)
    y.columns = ["besoin_n", "besoin_p", "besoin_k"]
    y = y.astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = np.clip(np.rint(pipeline.predict(X_test)), 0, None).astype(int)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mae", round(mae, 2))
    mlflow.log_metric("rmse", round(rmse, 2))
    mlflow.log_metric("r2", round(r2, 2))

    print("Registering the model via MLFlow...")

    conda_env = {
        'name': 'mlflow-env',
        'channels': ['conda-forge'],
        'dependencies': [
            'python=3.10.15',
            'pip<=21.3.1',
            {'pip': [
                'mlflow==2.17.0',
                'cloudpickle==2.2.1',
                'pandas==1.5.3',
                'psutil==5.8.0',
                'scikit-learn==1.5.2',
                'numpy==1.26.4',
            ]}
        ],
    }

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        registered_model_name=registered_model_name,
        artifact_path="model",
        conda_env=conda_env
    )

    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path=os.path.join("outputs", registered_model_name)
    )

    joblib.dump(pipeline, f"outputs/sklearn_model.joblib")

    mlflow.end_run()

    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--registered_model_name", type=str, default="terrabyte_model")
    args = parser.parse_args()

    print("Chargement du fichier :", args.input_data)
    df = pd.read_csv(args.input_data, sep=";")

    with mlflow.start_run():
        pipeline = create_pipeline()
        train_and_eval_model(pipeline, df, args.registered_model_name)

if __name__ == "__main__":
    main()
