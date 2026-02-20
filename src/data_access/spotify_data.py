import sys
import pandas as pd
import numpy as np
import logging
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME,COLLECTION_NAME
from src.exception import MyException
from src.logger import logging 

class SpotifyData:
    """
    This class handles data-related operations for the Spotify Hit Prediction project.
    It provides a connection to a MongoDB database and methods to export data as a pandas DataFrame.
    """
    
    def __init__(self):
        """
        Initializes the SpotifyData class by creating a connection to the MongoDB client.
        """
        try:
            # Initialize MongoDB client with the specified database name
            logging.info(f"Initializing MongoDB client for database: {DATABASE_NAME}")
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            logging.info("MongoDB client initialization successful.")
            
        except Exception as e:
            logging.error(f"Error during MongoDB client initialization: {e}")
            raise MyException(e, sys)
        
    def export_collection_as_dataframe(self, collection_name: str = COLLECTION_NAME, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports a MongoDB collection into a pandas DataFrame.

        Args:
            collection_name (str): The name of the collection to export. Defaults to the value in constants.
            database_name (Optional[str]): The name of the database. If None, the default database is used.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the data from the specified MongoDB collection.
        
        Raises:
            MyException: If an error occurs during the data fetching or processing.
        """
        try:
            logging.info(f"Starting to export collection '{collection_name}' as a DataFrame.")

            # Determine the database to connect to.
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
                logging.info(f"Using default database.")
            else:
                # Correcting the variable name to 'collection' instead of 'collection_name'
                collection = self.mongo_client.client[database_name][collection_name]
                logging.info(f"Using specified database: {database_name}.")
            
            # Fetch data from MongoDB
            logging.info("Fetching all documents from the collection...")
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Successfully fetched {len(df)} documents.")
            
            # Check if the DataFrame is empty.
            if df.empty:
                logging.warning("The fetched DataFrame is empty.")
                return df
                
            # Drop the '_id' column, which is an internal MongoDB identifier.
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
                logging.info("Dropped the default '_id' column from the DataFrame.")
                
            # Replace placeholder string "na" with numpy's NaN for proper numeric handling.
            df.replace({"na": np.nan}, inplace=True)
            logging.info("Replaced 'na' values with NaN.")
            
            return df

        except Exception as e:
            logging.error(f"Error exporting collection '{collection_name}' as DataFrame: {e}")
            raise MyException(e, sys)