import os
import pickle
from google.cloud import bigquery, storage
from dotenv import load_dotenv
from .modelconfig import MODELS, ModelConfig, ModelTrainer

class Training:
    def __init__(
        self,
        key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS",
        project_id="fleet-petal-448410-u6",
        dataset_id="titanic_dataset",
        preprocessed_table="preprocessed_data",
        bucket_name="titanic_model_2025_02_07"
    ):
        """
        Initialize the Training class.
        
        :param key_path_env_var: Environment variable that stores the service account key path.
        :param project_id: GCP project ID.
        :param dataset_id: BigQuery dataset containing the preprocessed table.
        :param preprocessed_table: Name of the preprocessed BigQuery table.
        :param bucket_name: GCS bucket where the trained model will be saved.
        """
        self._load_credentials(key_path_env_var)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.preprocessed_table = preprocessed_table
        self.bucket_name = bucket_name
        self.bq_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client()
        self.model = None

    def _load_credentials(self, key_path_env_var):
        """Load service account key using dotenv."""
        load_dotenv()  # Load .env file if available
        key_path = os.getenv(key_path_env_var)
        if not key_path or not os.path.exists(key_path):
            raise EnvironmentError(
                f"Service account key not found using '{key_path_env_var}'. Check your environment or .env file."
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        print(f"Using service account key from: {key_path}")

    def get_preprocessed_data(self):
        """
        Query the preprocessed data table from BigQuery and split into features (X) and target (y).
        
        :return: tuple (X, y) as pandas DataFrames.
        """
        table_full_name = f"{self.project_id}.{self.dataset_id}.{self.preprocessed_table}"
        query = f"SELECT * FROM `{table_full_name}`"
        query_job = self.bq_client.query(query)
        df = query_job.result().to_dataframe()
        if df.empty:
            print("No data found in the preprocessed table.")
        else:
            print(f"Preprocessed data loaded: shape {df.shape}")
        X = df.drop("Transported", axis=1)
        y = df["Transported"]
        return X, y

    def train_model(self, X, y, model_key="rf"):
        """
        Split the data, instantiate the model configuration and trainer,
        train the model, and report accuracy.
        
        :param X: Features DataFrame.
        :param y: Target Series.
        :param model_key: The key used to select the model configuration (default 'rf').
        :return: The trained model pipeline.
        """
        # Create the model configuration based on the provided model_key.
        model_config = ModelConfig(MODELS[model_key])
        
        # Instantiate the trainer using the configuration.
        trainer = ModelTrainer(model_config)
        
        # Train the model. Note: ModelTrainer.train_model() internally splits the data,
        # trains the pipeline, and prints accuracy.
        self.model = trainer.train_model(X, y)
        
        return self.model

    def save_model(self, model_key="rf"):
        """
        Save the trained model to GCS with versioning.
        
        The versioning scheme for a given model key:
          - If no model exists, save as "model_{model_key}.pkl".
          - If "model_{model_key}.pkl" exists, then save the new model as "model_{model_key}.v01.pkl".
          - Otherwise, if versioned files exist (e.g., model_{model_key}.v01.pkl), the new model
            will be named with an incremented version.
        
        :param model_key: A string identifier for the model (e.g., "rf").
        :return: The new model file name.
        """
        if self.model is None:
            raise ValueError("No trained model found. Train a model before saving.")

        bucket = self.storage_client.bucket(self.bucket_name)
        # Only list files that start with the expected prefix.
        blobs = list(bucket.list_blobs(prefix=f"model_{model_key}"))
        version_numbers = []

        base_filename = f"model_{model_key}.pkl"
        # Check existing model files
        for blob in blobs:
            name = blob.name
            if name == base_filename:
                version_numbers.append(0)
            elif name.startswith(f"model_{model_key}.v") and name.endswith(".pkl"):
                try:
                    version_str = name[len(f"model_{model_key}.v") : -len(".pkl")]
                    version_numbers.append(int(version_str))
                except ValueError:
                    pass

        # Decide on the new file name
        if not version_numbers:
            new_model_name = base_filename
        else:
            new_version = max(version_numbers) + 1
            new_model_name = f"model_{model_key}.v{new_version:02d}.pkl"
        
        # Save locally
        local_file = new_model_name
        with open(local_file, "wb") as f:
            pickle.dump(self.model, f)
        # Upload the file to GCS
        blob = bucket.blob(new_model_name)
        blob.upload_from_filename(local_file)
        print(f"Model uploaded to gs://{self.bucket_name}/{new_model_name}")
        return new_model_name

    def load_model(self, model_key="rf"):
        """
        Load the most recent model corresponding to the given model_key from GCS.

        It looks for model files that include the model_key in their filename and
        returns the one with the highest version.

        Expected filename patterns:
          - Base file: "model_{model_key}.pkl" (version 0)
          - Versioned file: "model_{model_key}.v{version_number:02d}.pkl"

        :param model_key: A string identifier for the model (e.g., "rf").
        :return: The loaded model, or None if no matching model is found.
        """
        bucket = self.storage_client.bucket(self.bucket_name)
        blobs = list(bucket.list_blobs(prefix=f"model_{model_key}"))

        if not blobs:
            print("No model files found in GCS.")
            return None

        latest_version = -1
        latest_blob_name = None
        base_filename = f"model_{model_key}.pkl"

        for blob in blobs:
            name = blob.name
            if name == base_filename:
                if latest_version < 0:
                    latest_version = 0
                    latest_blob_name = name
            elif name.startswith(f"model_{model_key}.v") and name.endswith(".pkl"):
                try:
                    version_str = name[len(f"model_{model_key}.v") : -len(".pkl")]
                    version_num = int(version_str)
                    if version_num > latest_version:
                        latest_version = version_num
                        latest_blob_name = name
                except ValueError:
                    pass

        if latest_blob_name is None:
            print(f"No valid model file found in GCS for model '{model_key}'.")
            return None

        # Download and load the latest model
        local_file = latest_blob_name
        blob = bucket.blob(latest_blob_name)
        blob.download_to_filename(local_file)
        with open(local_file, "rb") as f:
            self.model = pickle.load(f)
        print(f"Loaded model from gs://{self.bucket_name}/{latest_blob_name}")
        return self.model

if __name__ == "__main__":
    trainer = Training(key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS")
    
    # Query preprocessed data from BigQuery
    X, y = trainer.get_preprocessed_data()
    train_model_key = "rf"
    if not X.empty:
        # Train the model using the queried data
        trainer.train_model(X, y, train_model_key)
        # Save the model with versioning using the model key
        new_model_name = trainer.save_model(model_key=train_model_key)
        print(f"Model saved as: {new_model_name}")
        # Load the most recent model for the given model key
        loaded_model = trainer.load_model(model_key=train_model_key)
