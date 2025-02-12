import os

import numpy as np
import pandas as pd
from category_encoders import HashingEncoder
from dotenv import load_dotenv  # To handle .env file loading
from google.cloud import bigquery, storage
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Preprocess:
    def __init__(
        self,
        key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS",
        bucket_name="GCP_BUCKET",
        project_id="GCP_PROJECT_ID",
    ):
        """
        Initialize the Preprocess class.
        It sets up the connection to Google Cloud services and the preprocessing pipeline.

        :param key_path_env_var: str, name of the environment variable that stores the path to the service account key.
        :param bucket_name: str, name of the Google Cloud Storage bucket for saving/loading the preprocessor.
        """
        self.project_id = project_id
        self.key_path = self._get_key_path(key_path_env_var)
        self.bucket_name = bucket_name
        self.gcs_client = storage.Client()
        self.bq_client = bigquery.Client(project_id)
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
            raise EnvironmentError(
                f"Service account key not found in environment variable '{key_path_env_var}' or .env file."
            )

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        print(f"Using service account key from: {key_path}")
        return key_path

    def load_data(self, table_name) -> pd.DataFrame:
        """
        Load data from a BigQuery table.

        :param table_name: str, name of the table to query.
        :return: pd.DataFrame, the data loaded from BigQuery.
        """
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

    def preproc_data(self, df, output_table_name="preprocessed_data"):
        # Ensure no pandas.NA in the DataFrame
        df = df.astype(
            {
                col: "float" if df[col].dtype == "boolean" else df[col].dtype
                for col in df.columns
            }
        )

        X, y = df.drop("Transported", axis=1), df["Transported"]

        numeric_preprocessor = Pipeline(
            [
                (
                    "imputation_mean",
                    SimpleImputer(missing_values=np.nan, strategy="mean"),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        hashing_encoder = HashingEncoder(n_components=50)

        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(include=["object", "boolean"]).columns
        drop_cols = ["PassengerId", "Name", "Cabin"]

        self.preprocessor = ColumnTransformer(
            [
                ("categorical", hashing_encoder, cat_cols),
                ("numerical", numeric_preprocessor, num_cols),
                ("drop_cols", "drop", drop_cols),
            ]
        )

        # Set the output to pandas DataFrame format
        self.preprocessor.set_output(transform="pandas")

        X_processed = self.preprocessor.fit_transform(X)
        X_processed["Transported"] = y.values

        print("Preprocessed data ready. Uploading to BigQuery...")

        # Save the preprocessed DataFrame to BigQuery
        dataset_id = "titanic_dataset"
        table_id = f"{self.project_id}.{dataset_id}.{output_table_name}"

        job = self.bq_client.load_table_from_dataframe(
            X_processed,
            table_id,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        )
        job.result()  # Wait for the job to complete

        print(f"Preprocessed data saved to BigQuery table: {table_id} ✅")
        return table_id


# main
if __name__ == "__main__":
    preprocess = Preprocess(key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS")
    df = preprocess.load_data(table_name="RAW_train_data")
    if not df.empty:
        table_id = preprocess.preproc_data(df)
        print(f"Preprocessed data available at: {table_id}")
