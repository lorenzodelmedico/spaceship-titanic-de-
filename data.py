import os

#to install
import pickle
import pandas as pd
from google.cloud import bigquery, storage 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import numpy as np


# This function loads datas from the raw table in big query
def load_data(key_path, table_name) -> pd.DataFrame:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    project_id = 'fleet-petal-448410-u6'
    dataset_id = "titanic_dataset"
    table = table_name

    client = bigquery.Client(project=project_id)
    
    print(f"Processing: {table}")
    query = f"SELECT * FROM {dataset_id}.{table}"
    query_job = client.query(query)
    results = query_job.result()
    df = results.to_dataframe()
    if df is not None:
        print(f'Data loaded ✅ -> shape of the table {table}: {df.shape}')
        return df
    else: print(f'❌ No data in table {table}')

"""
this function use the loaded dataframe to preprocess data and save preprocessing to GCS if not already saved
If the preprocessing is already saved it loads the file from GCS, otherwise it upload a new file to GCS
It returns a tuple with X preprocessed and the target y
"""
def preproc_data(df) -> tuple:

    bucket_name = "titanic_model_2025_02_07"
    object_name = "preprocessor.pkl"
    local_file = "preprocessor.pkl"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    X, y = df.drop('Transported', axis=1), df['Transported']

    #writting the blob only if it doesn't exist
    if blob.exists() == False:
        numeric_preprocessor = Pipeline(
            steps=[
                ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_preprocessor = Pipeline(
            steps=[
                (
                    "imputation_mode",
                    SimpleImputer(missing_values=pd.NA,fill_value="missing", strategy="most_frequent"),
                ),
                ("onehot", OneHotEncoder(handle_unknown="error")),
            ]
        )

        num_cols = X.select_dtypes(include='number').columns
        cat_cols = X.select_dtypes(include=['object', 'boolean']).columns
        # Columns to be dropped
        drop_cols = ["PassengerId", "Name", "Cabin"]

        preprocessor = ColumnTransformer(
            [
                ("categorical", categorical_preprocessor, cat_cols),
                ("numerical", numeric_preprocessor, num_cols),
                ("drop_cols", "drop", drop_cols)
            ]
        )

        with open("preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
        print('Preprocessor saved localy 💾')

        blob.upload_from_filename(local_file)
        print(f"File uploaded to gs://{bucket_name}/{object_name} ✅")

    else:
        blob.download_to_filename(local_file)
        with open(local_file, "rb") as f:
            preprocessor = pickle.load(f)
        print(f'preprocessor loaded localy from GS: {bucket_name}/{object_name}')


    X_processed = preprocessor.fit_transform(X) 
    print(f'X processed and y separated ✅ ') 
    return X_processed, y 

if __name__ == "__main__":
    df = load_data(key_path= "./.env/key_sa_titanic_Hugo.json", table_name = 'RAW_train_data')
    X_processed, y = preproc_data(df)

