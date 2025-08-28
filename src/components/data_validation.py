import os,sys
from src.utils.main_utils import read_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants import SCHEMA_FILE_PATH
from src.logger import logging
from src.exception import MyException
import pandas as pd
import json
from abc import abstractmethod


class DataValidation:
    """
    This class is responsible for validating the ingested data based on a predefined schema.
    It checks for column existence and the number of columns in both training and testing datasets.
    """
    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        """
        Initializes the DataValidation class.

        Args:
            data_validation_config (DataValidationConfig): Configuration for data validation.
            data_ingestion_artifact (DataIngestionArtifact): Artifact from data ingestion, containing file paths.
        """
        try:
            logging.info(f"{'>>'*20} Data Validation Log Started {'<<'*20}")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise MyException(e,sys)
        
    
    def validate_number_of_columns(self,df:pd.DataFrame)->bool:
        """
        Validates the number of columns in the DataFrame against the schema.

        Args:
            df (pd.DataFrame): The DataFrame to be validated.

        Returns:
            bool: True if the number of columns matches the schema, False otherwise.
        """
        try:
            logging.info("Validating number of columns.")
            status=len(df.columns)==len(self._schema_config["columns"])
            logging.info(f"Number of columns validation status: {status}")
            return status
        except Exception as e:
            raise MyException(e,sys)
        
        
    def validate_column_existance(self,df:pd.DataFrame)->bool:
        """
        Validates the existence of all required columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be validated.

        Returns:
            bool: True if all required columns are present, False otherwise.
        """
        try:
            logging.info("Validating column existence.")
            required_columns=self._schema_config["columns"]
            present_columns=list(df.columns)
            missing_columns=[]
            for column in required_columns:
                if column not in present_columns:
                    missing_columns.append(column)
            logging.info(f"Missing columns are: {missing_columns}")
            
            return False if len(missing_columns)>0 else True
        
        except Exception as e:
            raise MyException(e,sys)
        
    @abstractmethod
    def load_data(file_path:str)->pd.DataFrame:
        """
        Loads data from a specified file path into a pandas DataFrame.

        Args:
            file_path (str): The path to the data file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            logging.info(f"Loading data from {file_path}")
            df=pd.read_csv(file_path)
            logging.info("Data loaded successfully.")
            return df
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_data_validation(self)->DataValidationArtifact:
        """
        Initiates the data validation process for both training and testing data.

        Returns:
            DataValidationArtifact: An artifact containing the validation status and report.
        """
        try:
            logging.info("Initiating data validation.")
            validation_msg=""
            train_df=DataValidation.load_data(file_path=self.data_ingestion_artifact.train_file_path)
            test_df=DataValidation.load_data(file_path=self.data_ingestion_artifact.test_file_path)
            
            # Validate number of columns for training data
            status=self.validate_number_of_columns(train_df)
            if not status:
                validation_msg +="Mismatch in number of columns in train set. "
            else:
                logging.info("Required number of columns are present in the training set.")
                
            # Validate number of columns for testing data
            status=self.validate_number_of_columns(test_df)
            if not status:
                validation_msg +="Mismatch in number of columns in test set. "
            else:
                logging.info("Required number of columns are present in the testing set.")
            
            # Validate column existence for training data
            status=self.validate_column_existance(train_df)
            if not status:
                validation_msg +="Missing columns in train set. "
            else:
                logging.info("Required columns are present in the training set.")
                
            # Validate column existence for testing data
            status=self.validate_column_existance(test_df)
            if not status:
                validation_msg +="Missing columns in test set. "
            else:
                logging.info("Required columns are present in the testing set.")
                
            validation_status=len(validation_msg)==0
            validation_artifact=DataValidationArtifact(
                validation_status=validation_status,
                message=validation_msg,
                report_file_path=self.data_validation_config.report_file_path,
                
            )
                
            report_dir=os.path.dirname(self.data_validation_config.report_file_path)
            os.makedirs(report_dir,exist_ok=True)
            
            validation_report={
                "validation_status": validation_status,
                "message": validation_msg.strip()
            }
                
            with open(self.data_validation_config.report_file_path,"w") as report_file:
                json.dump(validation_report,report_file,indent=4)
                
            logging.info(f"Data validation artifact: {validation_artifact}")
            logging.info(f"{'>>'*20} Data Validation Log Completed {'<<'*20}")

            return validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e