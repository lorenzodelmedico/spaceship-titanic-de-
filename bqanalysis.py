import os
from google.cloud import bigquery
from dotenv import load_dotenv

class BqAnalysis:
    def __init__(self, key_path_env_var="GOOGLE_APPLICATION_CREDENTIALS", bucket_name="titanic_model_2025_02_07", project_id="fleet-petal-448410-u6"):
        """
        Initialize the BqAnalysis class with the Google Cloud credentials.

        :param key_path_env_var: str, the environment variable name containing the service account key path.
        """
        self.key_path = self._get_key_path(key_path_env_var)
        self.bucket_name = bucket_name
        self.client = bigquery.Client(project_id)
    
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

    def list_datasets(self):
        """List all datasets in the project."""
        datasets = list(self.client.list_datasets())
        if datasets:
            print("\nDatasets in the project:")
            for dataset in datasets:
                print(f"  - {dataset.dataset_id}")
        else:
            print("\nNo datasets found.")

    def list_tables(self, dataset_id: str):
        """
        List all tables in a specified dataset.

        :param dataset_id: str, the dataset to list tables from.
        """
        tables = list(self.client.list_tables(dataset_id))
        if tables:
            print(f"\nTables in dataset {dataset_id}:")
            for table in tables:
                print(f"  - {table.table_id}")
        else:
            print(f"\nNo tables found in dataset {dataset_id}.")

    def get_table_info(self, dataset_id: str, table_id: str):
        """
        Get detailed information about a table, including row count, size, and schema.

        :param dataset_id: str, dataset containing the table.
        :param table_id: str, table to get information about.
        """
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = self.client.get_table(table_ref)

        print(f"\nTable: {dataset_id}.{table_id}")
        print(f"  - Number of rows: {table.num_rows}")
        print(f"  - Size: {table.num_bytes / (1024 ** 2):.2f} MB")
        print("  - Schema:")
        for schema_field in table.schema:
            print(f"    - {schema_field.name} ({schema_field.field_type})")

if __name__ == "__main__":
    # Initialize the class with the environment variable name where the key path is stored
    bq_analysis = BqAnalysis()
    
    # List all datasets
    bq_analysis.list_datasets()
    
    # List tables in a specific dataset
    dataset_id = "titanic_dataset"  # Replace with your dataset name
    bq_analysis.list_tables(dataset_id)
    
    # Get information about a specific table
    table_id = "RAW_train_data"  # Replace with your table name
    bq_analysis.get_table_info(dataset_id, table_id)
