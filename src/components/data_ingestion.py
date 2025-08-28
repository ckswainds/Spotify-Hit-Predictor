import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from src.constants import *
from src.data_access.spotify_data import SpotifyData


class DataIngestion:
    """
    DataIngestion Class:
    --------------------
    Handles fetching raw data from a source (e.g., MongoDB),
    splitting it into train/test sets, and saving them.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initializes the DataIngestion class with configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration object for data ingestion.
        """
        try:
            # Log the initialization of the DataIngestion class
            logging.info(f"{'>>' * 20} Starting Data Ingestion {'<<' * 20}")
            self.data_ingestion_config = data_ingestion_config
            logging.info("DataIngestion class initialized successfully.")
        except Exception as e:
            # Log the error and raise a custom exception
            logging.error("Error occurred during DataIngestion initialization.")
            raise MyException(e, sys)

    def get_data_from_server(self) -> pd.DataFrame:
        """
        Fetch data from a hypothetical external server.
        Note: This method is a placeholder and not implemented in this version.

        Returns:
            pd.DataFrame: A DataFrame containing the raw data.
        """
        logging.info("Fetching data from server (method not implemented yet).")
        pass

    def split_data_into_train_test(self, df: pd.DataFrame) -> None:
        """
        Splits the given DataFrame into training and testing sets and saves them as CSV files.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.
        """
        logging.info("Entered split_data_into_train_test method of DataIngestion class.")
        try:
            # Log the start of the data splitting process
            logging.info(f"Splitting dataset with test size: {self.data_ingestion_config.train_test_split_ratio}.")
            
            # Perform the train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                shuffle=True
            )
            logging.info("Data splitting completed successfully.")

            # Create the directory for saving the split data if it doesn't exist
            dir_name = os.path.dirname(self.data_ingestion_config.data_ingestion_train_file_path)
            os.makedirs(dir_name, exist_ok=True)
            logging.info(f"Created directory: {dir_name} (if not already present).")

            # Save the training and testing data to CSV files
            train_set.to_csv(self.data_ingestion_config.data_ingestion_train_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.data_ingestion_test_file_path, index=False, header=True)

            # Log the paths where the data was saved
            logging.info(f"Training data saved at {self.data_ingestion_config.data_ingestion_train_file_path}.")
            logging.info(f"Testing data saved at {self.data_ingestion_config.data_ingestion_test_file_path}.")

        except Exception as e:
            # Log the error and raise a custom exception with traceback
            logging.error("Error occurred while splitting data into train/test.")
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process by reading raw data from MongoDB,
        saving it, splitting it, and returning the paths as a DataIngestionArtifact.

        Returns:
            DataIngestionArtifact: An artifact object with paths to the training and testing data.
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class.")
        try:
            # Create a SpotifyData object to fetch data from MongoDB
            spotifydata = SpotifyData()
            # Export the collection to a DataFrame
            df = spotifydata.export_collection_as_dataframe()
            logging.info("Successfully fetched raw data from MongoDB.")

            # Create the directory for raw data storage
            raw_dir = os.path.dirname(self.data_ingestion_config.data_ingestion_raw_data_file)
            os.makedirs(raw_dir, exist_ok=True)
            
            # Save the raw data to a CSV file
            df.to_csv(self.data_ingestion_config.data_ingestion_raw_data_file, index=False, header=True)
            logging.info(f"Raw data loaded and saved at {self.data_ingestion_config.data_ingestion_raw_data_file}.")

            # Call the method to split the data
            self.split_data_into_train_test(df)

            # Create the DataIngestionArtifact object with the file paths
            data_ingestion_artifact = DataIngestionArtifact(
                raw_file_path=self.data_ingestion_config.data_ingestion_raw_data_file,
                train_file_path=self.data_ingestion_config.data_ingestion_train_file_path,
                test_file_path=self.data_ingestion_config.data_ingestion_test_file_path
            )

            # Log the completion of the process and the artifact details
            logging.info("Data Ingestion process completed successfully.")
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}.")

            return data_ingestion_artifact

        except Exception as e:
            # Log the error and raise a custom exception
            logging.error("Error occurred during initiate_data_ingestion process.")
            raise MyException(e, sys)