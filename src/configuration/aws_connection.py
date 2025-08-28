import boto3
import os
from src.constants import AWS_SECRET_ACCESS_KEY_ENV_KEY,AWS_ACCESS_KEY_ID_ENV_KEY,REGION_NAME
from src.logger import logging
from src.exception import MyException
import sys

class S3Client:
    """
    S3Client class to manage the connection to Amazon S3.
    This class is designed as a singleton to ensure only one client connection is established.
    It retrieves AWS credentials from environment variables for secure access.
    """
    s3_client=None
    s3_resource=None
    
    def __init__(self,region_name=REGION_NAME):
        """
        Initializes the S3Client by creating a single S3 resource and client.
        It checks for and validates AWS credentials from environment variables.
        """
        try:
            logging.info("Initializing S3Client.")
            if S3Client.s3_client==None or S3Client.s3_resource==None:
                logging.info("S3 clients not yet initialized. Retrieving credentials.")
                __access_key_id=os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
                __secret_access_key=os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
                
                if __access_key_id is None:
                    raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not set.")
                logging.info(f"Successfully retrieved {AWS_ACCESS_KEY_ID_ENV_KEY}.")

                if __secret_access_key is None:
                    raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")
                logging.info(f"Successfully retrieved {AWS_SECRET_ACCESS_KEY_ENV_KEY}.")
            
                logging.info("Creating S3 resource.")
                S3Client.s3_resource=boto3.resource(
                    "s3",
                    aws_access_key_id=__access_key_id,
                    aws_secret_access_key=__secret_access_key,
                    region_name=region_name
                )
                
                logging.info("Creating S3 client.")
                S3Client.s3_client = boto3.client('s3',
                                                aws_access_key_id=__access_key_id,
                                                aws_secret_access_key=__secret_access_key,
                                                region_name=region_name
                                                )
                
                self.s3_resource = S3Client.s3_resource
                self.s3_client = S3Client.s3_client
                logging.info("S3Client initialized successfully.")

        except Exception as e:
            logging.error("Error occurred during S3Client initialization.")
            raise MyException(e, sys)