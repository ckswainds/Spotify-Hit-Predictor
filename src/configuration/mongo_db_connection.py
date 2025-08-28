import os
import sys
import pymongo
import certifi
import logging

from src.exception import MyException
from src.logger import logging
from src.constants import *

# Get the path to the CA certificate file for TLS/SSL connections
ca = certifi.where()

class MongoDBClient:
    """
    A class to establish a singleton connection to a MongoDB database.
    This ensures that only one client connection is created across the application.
    """
    
    # Class attribute to hold the single client instance
    client = None
    
    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        """
        Initializes the MongoDB client connection.

        Args:
            database_name (str): The name of the MongoDB database to connect to.
        
        Raises:
            MyException: If the MongoDB URL environment variable is not set or if the connection fails.
        """
        try:
            # Check if a client instance already exists
            if MongoDBClient.client is None:
                # Log the attempt to establish a new connection
                logging.info("MongoDB client instance not found. Attempting to create a new connection.")

                # Get the MongoDB connection URL from an environment variable
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                
                # Check if the environment variable is set
                if mongo_db_url is None:
                    # Log the error and raise a custom exception
                    error_message = f"Environment variable '{MONGODB_URL_KEY}' is not set."
                    logging.error(error_message)
                    raise Exception(error_message)
                
                # Create the MongoDB client connection using the URL and TLS CA file
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                
            # Assign the singleton client instance to the object
            self.client = MongoDBClient.client
            # Set the database attribute by accessing it from the client
            self.database = self.client[database_name]
            # Store the database name for future reference
            self.database_name = database_name
            
            # Log successful connection
            logging.info(f"MongoDB connection to database '{self.database_name}' successful.")
                
        except Exception as e:
            # Log a more specific error message for the user's benefit
            logging.error(f"Failed to connect to MongoDB: {e}")
            # Raise a custom exception with traceback details
            raise MyException(e, sys)