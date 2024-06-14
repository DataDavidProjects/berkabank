from dataclasses import dataclass
import pandas as pd
from google.cloud import bigquery, storage
from typing import List, Optional
from pathlib import Path
from google.cloud.bigquery.job import LoadJobConfig, WriteDisposition
from jinja2 import Template  # pylint: disable=E0401
import urllib.parse
import os
from google.api_core.exceptions import Conflict, NotFound, GoogleAPICallError
from loguru import logger
from google.auth import default, exceptions
from google.oauth2 import service_account


def generate_query(input_file: Path, **replacements) -> str:
    """
    Read input file and replace placeholder using Jinja.

    Args:
        input_file (Path): input file to read
        replacements: keyword arguments to use to replace placeholders
    Returns:
        str: replaced content of input file
    """

    with open(input_file, "r", encoding="utf-8") as f:
        query_template = f.read()

    return Template(query_template).render(**replacements)


@dataclass
class TableConfig:
    """
    Configuration for creating a BigQuery table.

    Args:
        dataset_id (str): The ID of the dataset in BigQuery.
        table_id (str): The ID of the table in BigQuery.
        schema (List[bigquery.SchemaField], optional): The schema of the table. Defaults to None.
        df (pd.DataFrame, optional): The pandas DataFrame to create the table from. Defaults to None.
        partitioning_field (str, optional): The field to partition the table on. Defaults to None.
        partitioning_type (str, optional): The type of partitioning. Defaults to 'DAY'.
        partitioning_field_type (str, optional): The type of the partitioning field. Defaults to 'DATE'.
    """

    dataset_id: str
    table_id: str
    schema: List[bigquery.SchemaField] = None
    partitioning_field: str = None
    partitioning_type: str = "DAY"
    partitioning_field_type: str = "DATE"


@dataclass
class ExternalTableConfig:
    """
    Configuration for creating an external BigQuery table.

    Args:
        dataset_id (str): The ID of the dataset in BigQuery.
        table_id (str): The ID of the table in BigQuery.
        data_uri (str): The URI of the data in Google Cloud Storage.
        schema (List[bigquery.SchemaField]): The schema of the table.
        format (str, optional): The format of the data. Defaults to 'CSV'.
        skip_leading_rows (int, optional): The number of leading rows to skip. Defaults to 1.
    """

    dataset_id: str
    table_id: str
    data_uri: str
    schema: List[bigquery.SchemaField]
    format: str = "CSV"
    skip_leading_rows: int = 1


@dataclass
class BigQueryConf:
    """
    Configuration for BigQuery operations.

    Args:
        client (bigquery.Client): The BigQuery client.
    """

    client: Optional[bigquery.Client] = bigquery.Client()
    pipeline_name: str = "production"

    def __post_init__(self):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                f"pipelines/{self.pipeline_name}/service-account.json"
            )
            logger.info(
                f"Authenticated with Google Cloud SDK. for {self.pipeline_name}"
            )

            self.client = bigquery.Client(
                credentials=credentials, project=os.environ.get("PROJECT_ID")
            )
        except exceptions.DefaultCredentialsError:
            logger.error("Please authenticate with Google Cloud SDK.")
        except exceptions.GoogleAuthError:
            logger.error("Please authenticate with Google Cloud SDK.")
        except exceptions.RefreshError:
            logger.error("Please authenticate with Google Cloud SDK.")
        except exceptions.TransportError:
            logger.error("Please authenticate with Google Cloud SDK.")
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Cloud SDK: {e}")

    def create_dataset(
        self, dataset_id: str, location: str = "europe-west6"
    ) -> bigquery.Dataset:
        """
        Create a BigQuery dataset.

        Args:
            dataset_id (str): The ID of the dataset to create.

        Returns:
            bigquery.Dataset: The created dataset.
        """
        dataset_ref = self.client.dataset(dataset_id)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        dataset = self.client.create_dataset(dataset)  # API request
        print(f"Created dataset {dataset.project}.{dataset.dataset_id}")
        return dataset

    def create_table(self, config: TableConfig) -> bigquery.Table:
        """
        Create a table in BigQuery.

        Args:
            config (TableConfig): The configuration for the table.

        Returns:
            bigquery.Table: The created table.
        """
        table_ref = self.client.dataset(config.dataset_id).table(config.table_id)
        table = bigquery.Table(table_ref, schema=config.schema)
        if config.partitioning_field:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=config.partitioning_type,
                field=config.partitioning_field,
                require_partition_filter=True,
            )
        table = self.client.create_table(table)
        return table

    def create_table_from_pandas(
        self, df: pd.DataFrame, config: TableConfig
    ) -> bigquery.Table:
        """
        Create a table in BigQuery from a pandas DataFrame.

        Args:
            df (pd.DataFrame): The pandas DataFrame to create the table from.
            config (TableConfig): The configuration for the table.

        Returns:
            bigquery.Table: The created table.
        """
        if config.schema is None:
            pandas_dtype_to_bigquery_dtype = {
                "int64": "INT64",
                "float64": "FLOAT64",
                "bool": "BOOL",
                "datetime64[ns]": "TIMESTAMP",
                "object": "STRING",
            }
            config.schema = [
                bigquery.SchemaField(
                    name, pandas_dtype_to_bigquery_dtype[str(df[name].dtype)]
                )
                for name in df.columns
            ]
        table = self.create_table(config)
        job_config = bigquery.LoadJobConfig(schema=config.schema)
        job = self.client.load_table_from_dataframe(df, table, job_config=job_config)
        job.result()  # Wait for the job to complete
        return table

    def create_external_table(self, config: ExternalTableConfig) -> bigquery.Table:
        """
        Create an external table in BigQuery that references data in Google Cloud Storage.

        Args:
            config (ExternalTableConfig): The configuration for the external table.

        Returns:
            bigquery.Table: The created table.
        """
        # Construct a BigQuery table reference
        table_ref = self.client.dataset(config.dataset_id).table(config.table_id)
        # Create the external config
        external_config = bigquery.ExternalConfig(config.format)
        external_config.source_uris = [config.data_uri]
        external_config.schema = config.schema
        external_config.options.skip_leading_rows = config.skip_leading_rows
        # Create the table
        table = bigquery.Table(table_ref, schema=config.schema)
        table.external_data_configuration = external_config
        table = self.client.create_table(table)  # API request
        print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
        return table

    def extend_table(self, df: pd.DataFrame, config: TableConfig) -> bigquery.Table:
        """
        Append rows to an existing table in BigQuery from a pandas DataFrame.

        Args:
            df (pd.DataFrame): The pandas DataFrame to append to the table.
            config (TableConfig): The configuration for the table.

        Returns:
            bigquery.Table: The updated table.
        """
        table_ref = self.client.dataset(config.dataset_id).table(config.table_id)
        job_config = LoadJobConfig(
            schema=config.schema,
            write_disposition=WriteDisposition.WRITE_APPEND,
        )
        job = self.client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        job.result()  # Wait for the job to complete
        table = self.client.get_table(table_ref)  # Get the updated table
        return table

    def delete_table(self, config: TableConfig) -> None:
        """
        Delete a table in BigQuery.

        Args:
            config (TableConfig): The configuration for the table.
        """
        table_ref = self.client.dataset(config.dataset_id).table(config.table_id)

        self.client.delete_table(table_ref)

    def delete_tables_dataset(self, dataset_id: str) -> None:
        """
        Delete all tables in a dataset in BigQuery.

        Args:
            dataset_id (str): The ID of the dataset.
        """
        dataset_ref = self.client.dataset(dataset_id)
        tables = self.client.list_tables(dataset_ref)
        logger.info(f"Deleting tables in dataset {dataset_id}")
        for table in tables:
            logger.info(f"Deleting table {table.reference}...")
            self.client.delete_table(table.reference)

    def delete_dataset(self, dataset_id: str) -> None:
        """
        Delete a dataset in BigQuery.

        Args:
            dataset_id (str): The ID of the dataset.
        """
        dataset_ref = self.client.dataset(dataset_id)
        self.client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)

    def get_uris(self, bucket_name: str, folder: str) -> List[str]:
        """
        Get a list of all the files in a specific bucket and folder.

        Args:
            bucket_name (str): The name of the bucket.
            folder (str): The name of the folder.

        Returns:
            List[str]: A list of URIs of the files.
        """
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=folder)
        uris = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]
        return uris

    def create_cloudstorage_connection(self, uris: List[str], dataset_id: str) -> None:
        """
        Create tables external connection in BigQuery based on the given URIs.

        Args:
            uris (List[str]): The URIs of the tables in Cloud Storage.
            dataset_id (str): The dataset name.

        Returns:
            None
        """
        for uri in uris:
            # Parse the file name from the URI
            parsed_uri = urllib.parse.urlparse(uri)
            file_name = parsed_uri.path.strip("/").split("/")[-1]

            # Construct the table ID
            table_id = file_name.rsplit(".", 1)[0]
            logger.info(f"Parsed table_id:{table_id}")

            # Construct a BigQuery table reference
            table_ref = self.client.dataset(dataset_id).table(table_id)

            # Create the external config
            external_config = bigquery.ExternalConfig("CSV")
            external_config.source_uris = [uri]
            external_config.autodetect = True

            # Create the table
            table = bigquery.Table(table_ref)
            table.external_data_configuration = external_config
            try:
                table = self.client.create_table(table)  # API request
                table = self.client.get_table(table)  # Reload the table metadata
                logger.info(
                    f"Created table {table.project}.{table.dataset_id}.{table.table_id}"
                )
            except Conflict:
                # If the table already exists, update its external_data_configuration
                try:
                    table = self.client.get_table(table)  # Get the existing table
                    table.external_data_configuration = external_config
                    table = self.client.update_table(
                        table, ["external_data_configuration"]
                    )  # API request
                    logger.info(
                        f"Updated table {table.project}.{table.dataset_id}.{table.table_id}"
                    )
                except GoogleAPICallError as e:
                    # logger.error(f"Failed to update table due to API error: {e}")
                    pass
                except Exception as e:
                    logger.error(f"Failed to update table due to unexpected error: {e}")
            except GoogleAPICallError as e:
                logger.error(f"Failed to create table due to API error: {e}")
            except Exception as e:
                logger.error(f"Failed to create table due to unexpected error: {e}")
