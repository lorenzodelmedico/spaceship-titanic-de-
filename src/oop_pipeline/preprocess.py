import os

import numpy as np
import pandas as pd
from category_encoders import HashingEncoder
from dotenv import load_dotenv  # To handle .env file loading
from getenv import get_env_variable
from google.cloud import bigquery, storage
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Preprocess:
    def __init__(
        self,
        key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS",
        bucket_name_env_var="GCP_BUCKET",
        project_id_env_var="GCP_PROJECT_ID",
    ):
        """
        Initialize the Preprocess class.
        It sets up the connection to Google Cloud services and the preprocessing pipeline.

        :param key_path_env_var: str, name of the environment variable that stores the path to the service account key.
        :param bucket_name_env_var: str, name of the environment variable that stores the Google Cloud Storage bucket name.
        :param project_id_env_var: str, name of the environment variable that stores the GCP project ID.
        """
        load_dotenv()  # Load environment variables from .env file

        self.key_path = get_env_variable(key_path_env_var)
        self.bucket_name = get_env_variable(bucket_name_env_var)
        self.project_id = get_env_variable(project_id_env_var)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.key_path
        print(f"Using service account key from: {self.key_path}")

        self.gcs_client = storage.Client()
        self.bq_client = bigquery.Client(project=self.project_id)
        self.preprocessor = None

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
