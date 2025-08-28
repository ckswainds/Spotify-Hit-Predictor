from src.constants import *
import os
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP:str=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    artifact_dir:str=os.path.join(ARTIFACT_DIR,TIMESTAMP)
    timestamp:str=TIMESTAMP

train_pipeline_config=TrainingPipelineConfig()
@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str=os.path.join(train_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    data_ingestion_ingested_dir:str=os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR_NAME)
    data_ingestion_raw_data_file:str=os.path.join(data_ingestion_dir,"raw_data",DATA_INGESTION_RAW_DATA_FILE)
    data_ingestion_train_file_path:str=os.path.join(data_ingestion_ingested_dir,TRAIN_FILE_NAME)
    data_ingestion_test_file_path:str=os.path.join(data_ingestion_ingested_dir,TEST_FILE_NAME)
    train_test_split_ratio:float=TRAIN_TEST_SPLIT_RATIO
    
@dataclass
class DataValidationConfig:
    data_validation_dir:str=os.path.join(train_pipeline_config.artifact_dir,DATA_VALIDATION_DIR)
    report_file_path:str=os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_FILE_NAME)
    
@dataclass
class DataTransformationConfig:
    data_transformation_dir:str=os.path.join(train_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR)
    transformed_train_file_path:str=os.path.join(data_transformation_dir,TRANSFORMED_DATA_DIR,TRANSFORMED_TRAIN_FILE)
    transformed_test_file_path:str=os.path.join(data_transformation_dir,TRANSFORMED_DATA_DIR,TRANSFORMED_TEST_FILE)
    # train_label_file_path:str=os.path.join(data_transformation_dir,TRANSFORMED_DATA_DIR,TRAIN_LABEL_FILE)
    # test_label_file_path:str=os.path.join(data_transformation_dir,TRANSFORMED_DATA_DIR,TEST_LABEL_FILE)
    transformed_object_file_path:str=os.path.join(data_transformation_dir,TRANSFORMED_OBJECT_DIR,PREPROCESSOR_OBJECT_FILE)
    
    
@dataclass 
class ModelTrainerConfig:
    model_trainer_dir:str=os.path.join(train_pipeline_config.artifact_dir,MODEL_TRAINER_DIR)
    trained_model_file_path:str=os.path.join(model_trainer_dir,MODEL_NAME)
    train_model_report_dir:str=os.path.join(model_trainer_dir,TRAINED_MODEL_REPORT_DIR)
    trained_model_report_file_path=os.path.join(train_model_report_dir,TRAINED_MODEL_REPORT_FILE)
    
    
@dataclass
class ModelEvaluationConfig:
    changed_threshold_score:float=MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name:str=MODEL_BUCKET_NAME
    s3_model_key_path=MODEL_NAME
 
@dataclass
class ModelPusherConfig:
    bucket_name:str=MODEL_BUCKET_NAME
    s3_model_key_path=MODEL_NAME   
    
    
    
@dataclass 
class SpotifyHitPredictorConfig:
    model_file_name:str=MODEL_NAME
    model_bucket_name:str=MODEL_BUCKET_NAME
    
