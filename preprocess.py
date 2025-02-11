import os
import pickle
import pandas as pd
import numpy as np
from google.cloud import bigquery, storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv  # To handle .env file loading

class Preprocess:
    def __init__(self, key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS", bucket_name="titanic_model_2025_02_07"):
        """
        Initialize the Preprocess class.
        It sets up the connection to Google Cloud services and the preprocessing pipeline.
        
        :param key_path_env_var: str, name of the environment variable that stores the path to the service account key.
        :param bucket_name: str, name of the Google Cloud Storage bucket for saving/loading the preprocessor.
        """
        self.key_path = self._get_key_path(key_path_env_var)
        self.bucket_name = bucket_name
        self.gcs_client = storage.Client()
        self.bq_client = bigquery.Client()
        self.preprocessor = None

    def _get_key_path(self, key_path_env_var):
        """
        Get the service account key path. Check system environment first; if not found, load from .env file.
        
        :param key_path_env_var: str, environment variable name for the key path.
        :return: str, the path to the service account key.
        """
        key_path = os.getenv(key_path_env_var)
        if not key_path:
            load_dotenv()  # Load .env if key is not in system environment
            key_path = os.getenv(key_path_env_var)
        
        if not key_path:
            raise EnvironmentError(f"Service account key not found in environment variable '{key_path_env_var}' or .env file.")
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        print(f"Using service account key from: {key_path}")
        return key_path

    def load_data(self, table_name) -> pd.DataFrame:
        """
        Load data from a BigQuery table.
        
        :param table_name: str, name of the table to query.
        :return: pd.DataFrame, the data loaded from BigQuery.
        """
        project_id = "fleet-petal-448410-u6"
        dataset_id = "titanic_dataset"
        
        query = f"SELECT * FROM {dataset_id}.{table_name}"
        query_job = self.bq_client.query(query)
        results = query_job.result()
        df = results.to_dataframe()
        
        if not df.empty:
            print(f"Data loaded ✅ -> shape of the table {table_name}: {df.shape}")
            return df
        else:
            print(f"❌ No data in table {table_name}")
            return pd.DataFrame()

    def preproc_data(self, df) -> tuple:
        """
        Preprocess the data and save/load the preprocessor to/from Google Cloud Storage.
        
        :param df: pd.DataFrame, the input data.
        :return: tuple, preprocessed X and target y.
        """
        object_name = "preprocessor.pkl"
        local_file = "preprocessor.pkl"
        
        bucket = self.gcs_client.bucket(self.bucket_name)
        blob = bucket.blob(object_name)
        
        X, y = df.drop('Transported', axis=1), df['Transported']
        
        if not blob.exists():
            print("Preprocessor not found in GCS. Creating a new one...")
            numeric_preprocessor = Pipeline([
                ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
                ("scaler", StandardScaler())
            ])
            
            categorical_preprocessor = Pipeline([
                ("imputation_mode", SimpleImputer(missing_values=pd.NA, fill_value="missing", strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="error"))
            ])
            
            num_cols = X.select_dtypes(include='number').columns
            cat_cols = X.select_dtypes(include=['object', 'boolean']).columns
            drop_cols = ["PassengerId", "Name", "Cabin"]
            
            self.preprocessor = ColumnTransformer([
                ("categorical", categorical_preprocessor, cat_cols),
                ("numerical", numeric_preprocessor, num_cols),
                ("drop_cols", "drop", drop_cols)
            ])
            
            with open(local_file, "wb") as f:
                pickle.dump(self.preprocessor, f)
            print("Preprocessor saved locally 💾")
            
            blob.upload_from_filename(local_file)
            print(f"File uploaded to gs://{self.bucket_name}/{object_name} ✅")
        
        else:
            blob.download_to_filename(local_file)
            with open(local_file, "rb") as f:
                self.preprocessor = pickle.load(f)
            print(f"Preprocessor loaded locally from GCS: {self.bucket_name}/{object_name}")
        
        X_processed = self.preprocessor.fit_transform(X)
        print("X processed and y separated ✅")
        return X_processed, y

# Example usage
if __name__ == "__main__":
    preprocess = Preprocess(key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS")
    df = preprocess.load_data(table_name="RAW_train_data")
    if not df.empty:
        X_processed, y = preprocess.preproc_data(df)
