from dotenv import load_dotenv
from getenv import get_env_variable
from google.cloud import bigquery, storage


class BqAnalysis:
    def __init__(
        self,
        key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS",
        project_id="GCP_PROJECT_ID",
    ):
        """
        Initialize the BqAnalysis class with credentials, BigQuery client, and Storage client.

        :param key_path_env_var: Environment variable name that stores the service account key path.
        :param project_id: GCP project ID.
        """
        load_dotenv()
        self.key_path_env_var = get_env_variable(key_path_env_var)
        self.project_id = get_env_variable(project_id)
        self.bq_client = bigquery.Client(project=self.project_id)
        self.storage_client = storage.Client()

    def list_datasets(self):
        """
        List all datasets in the project.
        """
        datasets = list(self.bq_client.list_datasets())
        if datasets:
            print("\nDatasets in the project:")
            for dataset in datasets:
                print(f"  - {dataset.dataset_id}")
        else:
            print("No datasets found.")

    def list_tables(self, dataset_id):
        """
        List all tables in the specified dataset.

        :param dataset_id: Name of the dataset.
        """
        tables = list(self.bq_client.list_tables(dataset_id))
        if tables:
            print(f"\nTables in dataset '{dataset_id}':")
            for table in tables:
                print(f"  - {table.table_id}")
        else:
            print(f"No tables found in dataset '{dataset_id}'.")

    def get_table_info(self, dataset_id, table_id):
        """
        Get detailed information about a specific table.

        :param dataset_id: Name of the dataset.
        :param table_id: Name of the table.
        """
        table_ref = self.bq_client.dataset(dataset_id).table(table_id)
        table = self.bq_client.get_table(table_ref)
        print(f"\nTable: {dataset_id}.{table_id}")
        print(f"  - Number of rows: {table.num_rows}")
        print(f"  - Size: {table.num_bytes / (1024 ** 2):.2f} MB")
        print("  - Schema:")
        for field in table.schema:
            print(f"    - {field.name} ({field.field_type})")

    def list_pkl_files_in_bucket(self, bucket_name):
        """
        List all .pkl files saved in the specified GCS bucket.

        :param bucket_name: Name of the GCS bucket.
        """
        bucket = self.storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs())
        pkl_files = [blob.name for blob in blobs if blob.name.endswith(".pkl")]
        if pkl_files:
            print(f"\nList of .pkl files in bucket '{bucket_name}':")
            for file in pkl_files:
                print(f"  - {file}")
        else:
            print(f"\nNo .pkl files found in bucket '{bucket_name}'.")


if __name__ == "__main__":
    # Initialize the BqAnalysis class (ensure your .env or system environment has the GCP_KEY_PATH variable)
    analysis = BqAnalysis(key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS")

    # List datasets
    analysis.list_datasets()

    # List tables in a specific dataset
    dataset = "titanic_dataset"
    analysis.list_tables(dataset)

    # Get information about a specific table
    analysis.get_table_info(dataset, "RAW_train_data")

    # List .pkl files in a specified bucket (for example, where model files are saved)
    bucket = "titanic_model_2025_02_07"
    analysis.list_pkl_files_in_bucket(bucket)
