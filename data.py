import os

#to install
import pickle
import pandas as pd
from google.cloud import bigquery, storage 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from dotenv import load_dotenv

def load_credentials(key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS"):
    """
    Load the service account key from an environment variable or a .env file.
    """
    load_dotenv()  # Load .env file if available
    key_path = os.getenv(key_path_env_var)
    if not key_path or not os.path.exists(key_path):
        raise EnvironmentError(f"Service account key not found in environment variable '{key_path_env_var}' or .env file.")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    print(f"Using service account key from: {key_path}")
    
# This function loads datas from the raw table in big query
def load_data(key_path, table_name) -> pd.DataFrame:
    load_credentials(key_path)
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
def preproc_data(df, training=True):

    bucket_name = "titanic_model_2025_02_07"
    object_name = "preprocessor.pkl"
    local_file = "preprocessor.pkl"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    drop_cols = ["PassengerId", "Name", "Cabin"]
    # Separate features and target only during training
    if training:
        X, y = df.drop('Transported', axis=1), df['Transported']
    else:
        X = df  # No target variable during prediction

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
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        num_cols = X.select_dtypes(include='number').columns
        cat_cols = X.select_dtypes(include=['object', 'boolean']).columns
        # Columns to be dropped

        preprocessor = ColumnTransformer(
            [
                ("categorical", categorical_preprocessor, cat_cols),
                ("numerical", numeric_preprocessor, num_cols),
                ("drop_cols", "drop", drop_cols)
            ]
        )

        preprocessor.fit(X)

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
    # Ensure missing columns (those in drop_cols) are in the input data
    missing_cols = set(drop_cols) - set(X.columns)
    for col in missing_cols:
        X[col] = None
    #Here we use fit_transform for training (when we want to keep mean, std and mode parameters)
    #Otherwise we want to use transform when it is just for preprocessing datas
    X_processed = preprocessor.fit_transform(X) if training else preprocessor.transform(X)
    print('X processed and y separated ✅') if training else print('X processed ✅')
    return (X_processed, y) if training else X_processed

if __name__ == "__main__":
    df = load_data(key_path="GOOGLE_APPLICATION_CREDENTIALS", table_name = 'RAW_train_data')
    X_processed, y = preproc_data(df)

