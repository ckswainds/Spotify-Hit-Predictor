import os
from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    raw_file_path:str
    train_file_path:str
    test_file_path:str
    
    
@dataclass
class DataValidationArtifact:
    validation_status:bool
    message:str
    report_file_path:str
    
    
@dataclass
class DataTransformationArtifact:
    preprocessor_object_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str
    # train_label_dir:str
    # test_label_dir:str
@dataclass
class ClassificationMetricArtifact:
    accuracy_score:float
    precision_score:float
    recall_score:float
@dataclass
class ModelTrainerArtifact:
    trained_model_path:str
    trained_model_scores:ClassificationMetricArtifact
    
    
@dataclass 
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str
    trained_model_path:str


@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str