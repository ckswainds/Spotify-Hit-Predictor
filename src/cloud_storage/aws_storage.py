import boto3
from src.configuration.aws_connection import S3Client
from io import StringIO
from typing import Union,List
import os,sys
from src.logger import logging
from mypy_boto3_s3.service_resource import Bucket
from src.exception import MyException
from botocore.exceptions import ClientError
from pandas import DataFrame,read_csv
import pickle

class SimpleStorageService:
    """
    A class to interact with Amazon S3 for common operations like
    uploading files, reading objects, and managing buckets.
    """
    
    def __init__(self):
        """
        Initializes the SimpleStorageService with S3 client and resource.
        """
        logging.info(f"{'>>'*20} SimpleStorageService Log Started {'<<'*20}")
        s3_client=S3Client()
        self.s3_client=s3_client.s3_client
        self.s3_resource=s3_client.s3_resource
        
    def s3_key_path_available(self,bucket_name:str,s3_key:str)->bool:
        """
        Checks if a file or folder with the given key exists in the specified S3 bucket.

        Args:
            bucket_name (str): The name of the S3 bucket.
            s3_key (str): The key (path) to the object or folder.

        Returns:
            bool: True if the key path is available, False otherwise.
        """
        logging.info(f"Checking if S3 key path '{s3_key}' is available in bucket '{bucket_name}'.")
        try:
            bucket=self.get_bucket(bucket_name)
            file_objects=[file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            is_available = len(file_objects)>0
            logging.info(f"S3 key path availability: {is_available}")
            return is_available
        except Exception as e:
            raise MyException(e,sys)
        
        
    def get_bucket(self,bucket_name:str)->Bucket:
        """
        Retrieves a bucket resource from S3.

        Args:
            bucket_name (str): The name of the bucket to retrieve.

        Returns:
            Bucket: The S3 bucket resource.
        """
        try:
            logging.info(f"Getting bucket '{bucket_name}'.")
            bucket=self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of SimpleStorageService class")
            return bucket
        except Exception as e:
            raise MyException(e, sys) from e
        
    @staticmethod
    def read_object(object_name:object,decode:bool=True,make_readable:bool=False)->Union[StringIO,str]:
        """
        Reads a file object from S3.

        Args:
            object_name (object): The S3 object to read.
            decode (bool): If True, decodes the object content.
            make_readable (bool): If True, returns the content as a StringIO object.

        Returns:
            Union[StringIO,str]: The content of the object as a string or StringIO object.
        """
        logging.info("Entered the read_object method of SimpleStorageService class")
        try:
            content = object_name.get()["Body"].read()
            if decode:
                content = content.decode('utf-8')
            
            if make_readable:
                return StringIO(content)
            else:
                return content
        except Exception as e:
            raise MyException(e, sys) from e
        
    def get_file_object(self,filename:str,bucket_name:str)->Union[List[object],object]:
        """
        Retrieves S3 file objects with a given filename prefix.

        Args:
            filename (str): The filename or prefix to filter by.
            bucket_name (str): The name of the S3 bucket.

        Returns:
            Union[List[object],object]: A single file object if only one is found, otherwise a list of objects.
        """
        logging.info(f"Getting file object for filename '{filename}' from bucket '{bucket_name}'.")
        try:
            bucket=self.get_bucket(bucket_name=bucket_name)
            file_objects=[file_object for file_object in bucket.objects.filter(Prefix=filename)]
            
            if len(file_objects) == 1:
                return file_objects[0]
            else:
                return file_objects
        except Exception as e:
            raise MyException(e, sys) from e
    
    
    def load_model(self,model_name:str,bucket_name:str,model_dir:str=None)->object:
        """
        Loads a pickled model from an S3 bucket.

        Args:
            model_name (str): The name of the model file.
            bucket_name (str): The name of the S3 bucket.
            model_dir (str, optional): The directory (key prefix) where the model is stored. Defaults to None.

        Returns:
            object: The unpickled model object.
        """
        logging.info(f"Loading model '{model_name}' from bucket '{bucket_name}'.")
        try:
            model_file = os.path.join(model_dir, model_name) if model_dir else model_name
            file_object=self.get_file_object(model_file,bucket_name)
            model_obj=self.read_object(file_object,decode=False)
            model=pickle.loads(model_obj)
            logging.info("Production model loaded from S3 bucket.")
            return model
        except Exception as e:
            raise MyException(e, sys) from e
        
    def create_folder(self,folder_name:str,bucket_name:str)->None:
        """
        Creates a "folder" (a zero-byte object with a trailing slash) in an S3 bucket.

        Args:
            folder_name (str): The name of the folder to create.
            bucket_name (str): The name of the S3 bucket.
        """
        logging.info(f"Entered the create_folder method of SimpleStorageService class. Creating folder '{folder_name}' in bucket '{bucket_name}'.")
        try:
            folder_obj=folder_name+"/"
            self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
            logging.info(f"Successfully created folder '{folder_name}' in bucket '{bucket_name}'.")
        except ClientError as ce:
            logging.error(f"Failed to create folder. Error: {ce}")
            if ce.response["Error"]["Code"] == "404":
                raise MyException("Bucket not found.", sys) from ce
            else:
                raise MyException(ce, sys) from ce
        except Exception as e:
            raise MyException(e, sys) from e
        finally:
            logging.info("Exited the create_folder method of SimpleStorageService class")
            
    def upload_file(self,from_filename:str,to_filename:str,bucket_name:str,remove:bool=True):
        """
        Uploads a local file to a specified S3 bucket.

        Args:
            from_filename (str): The local path of the file to upload.
            to_filename (str): The target path (key) in the S3 bucket.
            bucket_name (str): The name of the S3 bucket.
            remove (bool): If True, removes the local file after a successful upload.
        """
        logging.info("Entered the upload_file method of SimpleStorageService class")
        try:
            logging.info(f"Uploading {from_filename} to {to_filename} in {bucket_name}")
            self.s3_resource.meta.client.upload_file(from_filename,bucket_name,to_filename)
            logging.info(f"Uploaded {from_filename} to {to_filename} in {bucket_name}")

            # Delete the local file if remove is True
            if remove:
                os.remove(from_filename)
                logging.info(f"Removed local file {from_filename} after upload")
            logging.info("Exited the upload_file method of SimpleStorageService class")
        except Exception as e:
            raise MyException(e, sys) from e
        
    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str) -> None:
        """
        Uploads a DataFrame as a CSV file to the specified S3 bucket.

        Args:
            data_frame (DataFrame): DataFrame to be uploaded.
            local_filename (str): Temporary local filename for the DataFrame.
            bucket_filename (str): Target filename in the bucket.
            bucket_name (str): Name of the S3 bucket.
        """
        logging.info("Entered the upload_df_as_csv method of SimpleStorageService class")
        try:
            logging.info(f"Saving DataFrame to local file '{local_filename}' for upload.")
            data_frame.to_csv(local_filename, index=None, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
            logging.info("Exited the upload_df_as_csv method of SimpleStorageService class")
        except Exception as e:
            raise MyException(e, sys) from e
            
    def get_df_from_object(self,object_:object)->DataFrame:
        """
        Reads an S3 object and returns its content as a pandas DataFrame.

        Args:
            object_ (object): The S3 file object to read.

        Returns:
            DataFrame: The DataFrame created from the object content.
        """
        logging.info("Entered the get_df_from_object method of SimpleStorageService class")
        try:
            logging.info("Reading object content and converting to DataFrame.")
            content=self.read_object(object_,make_readable=True)
            df=read_csv(content,na_values="na")
            logging.info("Exited the get_df_from_object method of SimpleStorageService class")
            return df
        except Exception as e:
            raise MyException(e, sys) from e
            
    def read_csv(self,filename:str,bucket_name:str)->DataFrame:
        """
        Reads a CSV file from an S3 bucket and returns it as a DataFrame.

        Args:
            filename (str): The name of the CSV file.
            bucket_name (str): The name of the S3 bucket.

        Returns:
            DataFrame: The DataFrame created from the CSV file.
        """
        logging.info("Entered the read_csv method of SimpleStorageService class")
        try:
            csv_obj=self.get_file_object(filename,bucket_name)
            df=self.get_df_from_object(csv_obj)
            logging.info("Exited the read_csv method of SimpleStorageService class")
            return df
        except Exception as e:
            raise MyException(e, sys) from e
