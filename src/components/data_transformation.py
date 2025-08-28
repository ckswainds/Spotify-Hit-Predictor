import os,sys
import pandas as pd
import numpy as np
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from src.utils.main_utils import read_yaml_file
import pickle 
from src.constants import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from abc import abstractmethod

class DataTransformation:
    """
    This class is responsible for the data transformation stage of the pipeline.
    It performs preprocessing steps like dropping columns, one-hot encoding categorical features,
    and scaling numerical features.
    """
    def __init__(self,data_transformation_config:DataTransformationConfig,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact):
        """
        Initializes the DataTransformation class.

        Args:
            data_transformation_config (DataTransformationConfig): Configuration for data transformation.
            data_ingestion_artifact (DataIngestionArtifact): Artifact from data ingestion.
            data_validation_artifact (DataValidationArtifact): Artifact from data validation.
        """
        try:
            logging.info(f"{'>>'*20} Data Transformation Log Started {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    def drop_columns(self,train_df:pd.DataFrame,test_df:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
        """
        Drops the columns specified in the schema from both the training and testing DataFrames.

        Args:
            train_df (pd.DataFrame): The training DataFrame.
            test_df (pd.DataFrame): The testing DataFrame.

        Returns:
            tuple[pd.DataFrame,pd.DataFrame]: The DataFrames after dropping the specified columns.
        """
        try:
            logging.info("Dropping columns specified in schema.")
            train_df.drop(columns=self._schema_config["columns_to_drop"],axis=1,inplace=True)
            test_df.drop(columns=self._schema_config["columns_to_drop"],axis=1,inplace=True)
            logging.info("Columns dropped successfully.")
            return train_df,test_df
        except Exception as e:
            raise MyException(e,sys)
        
    def save_trasformed_data_and_object(self,train_df:pd.DataFrame,test_df:pd.DataFrame)->None:
        """
        Applies transformations to the data, saves the transformed data as numpy arrays,
        and saves the preprocessor object using pickle.
        
        Args:
            train_df (pd.DataFrame): The training DataFrame.
            test_df (pd.DataFrame): The testing DataFrame.
        """
        try:
            logging.info("Splitting data into features and target.")
            numerical_features=self._schema_config["numerical_features"]
            categorical_features=self._schema_config["categorical_features"]
            
            X_train,y_train=train_df.drop(self._schema_config["target_column"],axis=1),train_df[self._schema_config["target_column"]]
            X_test,y_test=test_df.drop(self._schema_config["target_column"],axis=1),test_df[self._schema_config["target_column"]]
            
            # Creating preprocessing pipelines for numerical and categorical features
            logging.info("Creating preprocessing pipelines.")
            numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
             ])

            # Categorical pipeline
            categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            # Creating ColumnTransformer to apply pipelines to respective columns
            preprocessor=ColumnTransformer(transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
            ],remainder="passthrough")
            
            # Fitting and transforming the data
            logging.info("Fitting preprocessor on training data and transforming both train and test data.")
            X_train_transformed=preprocessor.fit_transform(X_train)
            X_test_transformed=preprocessor.transform(X_test)
            
            # Saving the preprocessor object
            logging.info("Saving preprocessor object.")
            transformed_obj_dir=os.path.dirname(self.data_transformation_config.transformed_object_file_path)
            os.makedirs(transformed_obj_dir, exist_ok=True)
            with open(self.data_transformation_config.transformed_object_file_path,"wb") as f:
                pickle.dump(preprocessor,f)
            
            # Saving transformed data as numpy arrays
            transformed_data_dir=os.path.dirname(self.data_transformation_config.transformed_train_file_path)
            os.makedirs(transformed_data_dir,exist_ok=True)
            logging.info("Saving transformed training and testing data.")
            np.savez(self.data_transformation_config.transformed_train_file_path, X=X_train_transformed, y=y_train)
            np.savez(self.data_transformation_config.transformed_test_file_path, X=X_test_transformed, y=y_test)

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
            
    def initiate_data_transformation(self)->DataTransformationArtifact:
        """
        Initiates the data transformation process.
        
        Raises:
            Exception: If the data validation status is not True.
        
        Returns:
            DataTransformationArtifact: An artifact containing paths to the transformed data and preprocessor object.
        """
        try:
            logging.info("Initiating data transformation.")
            status=self.data_validation_artifact.validation_status
            if not status:
                logging.info("Data validation status is false. Please check and validate data.")
                raise Exception(self.data_validation_artifact.message)
            
            logging.info("Loading training and testing data for transformation.")
            train_df,test_df=(DataTransformation.load_data(self.data_ingestion_artifact.train_file_path),
                              DataTransformation.load_data(self.data_ingestion_artifact.test_file_path))
            
            # Dropping columns and saving transformed data and object
            train_df,test_df=self.drop_columns(train_df=train_df,test_df=test_df)
            self.save_trasformed_data_and_object(train_df=train_df,test_df=test_df)
            
            # Creating and returning the data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                preprocessor_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            logging.info(f"{'>>'*20} Data Transformation Log Completed {'<<'*20}")
            
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e,sys)
