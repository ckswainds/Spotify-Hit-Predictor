import sys
from src.logger import logging
from src.entity.artifact_entity import ModelEvaluationArtifact,DataTransformationArtifact,ModelTrainerArtifact,DataIngestionArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.s3_estimator import Proj1Estimator
from sklearn.metrics import accuracy_score
from typing import Optional
from src.exception import MyException
import numpy as np
import pandas as pd
from src.constants import SCHEMA_FILE_PATH
from src.utils.main_utils import read_yaml_file
class ModelEvaluation:
    """
    This class is responsible for evaluating the newly trained model against
    the existing production model stored in S3. It decides whether to accept
    the new model based on a performance comparison.
    """
    def __init__(self,model_evaluation_config:ModelEvaluationConfig,
                 model_trainer_artifact:ModelTrainerArtifact,
                 data_transformation_artifact:DataTransformationArtifact,
                 data_ingestion_artifact:DataIngestionArtifact):
        """
        Initializes the ModelEvaluation class.

        Args:
            model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation.
            model_trainer_artifact (ModelTrainerArtifact): The artifact from the model training stage.
            data_transformation_artifact (DataTransformationArtifact): The artifact from the data transformation stage.
        """
        try:
            logging.info(f"{'>>'*20} Model Evaluation Log Started {'<<'*20}")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact=data_ingestion_artifact
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
            
            
            logging.info("Loading transformed test data.")
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df.drop(columns=self._schema_config["columns_to_drop"],axis=1,inplace=True)
            self.X_test,self.y_test=test_df.drop(self._schema_config["target_column"],axis=1),test_df[self._schema_config["target_column"]]
            
            logging.info("Test data loaded successfully.")
        
        except Exception as e:
            raise MyException(e,sys)
        
        
    def get_best_model(self)->Optional[Proj1Estimator]:
        """
        Retrieves the best model from the S3 bucket if it exists.

        Returns:
            Optional[Proj1Estimator]: The Proj1Estimator object if a model is found, otherwise None.
        """
        try:
            logging.info("Fetching the best model from S3.")
            proj1_estimator = Proj1Estimator(bucket_name=self.model_evaluation_config.bucket_name,
                                             model_path=self.model_evaluation_config.s3_model_key_path)
            
            if proj1_estimator.is_model_present(self.model_evaluation_config.s3_model_key_path):
                logging.info("Best model found in S3.")
                return proj1_estimator
            
            logging.info("No best model found in S3. Proceeding with evaluation against no model.")
            return None
        except Exception as e:
            raise MyException(e,sys)
        
    def evaluate_model(self)->ModelEvaluationArtifact:
        """
        Compares the trained model's accuracy with the production model's accuracy.

        Returns:
            ModelEvaluationArtifact: An artifact containing the evaluation results.
        """
        try:
            logging.info("Starting model evaluation.")
            trained_model_score = self.model_trainer_artifact.trained_model_scores.accuracy_score
            best_model_score = None
            best_model = self.get_best_model()
            
            if best_model is not None:
                logging.info("Best model from S3 found. Evaluating its performance.")
                y_preds_best_model = best_model.predict(self.X_test)
                best_model_score = accuracy_score(self.y_test, y_preds_best_model)
                logging.info(f"Best model accuracy: {best_model_score}")
            else:
                logging.info("No existing production model to compare against.")

            tmp_best_score = 0 if best_model_score is None else best_model_score
            is_model_accepted = trained_model_score > tmp_best_score
            score_difference = trained_model_score - tmp_best_score
            
            logging.info(f"Trained model accuracy: {trained_model_score}")
            logging.info(f"Accuracy difference: {score_difference}")
            logging.info(f"Is new model accepted? {is_model_accepted}")

            evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                s3_model_path=self.model_evaluation_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_path,
                changed_accuracy=score_difference
            )
            
            logging.info(f"Model evaluation completed. Artifact: {evaluation_artifact}")
            return evaluation_artifact

        except Exception as e:
            logging.error("Error occurred during model evaluation.")
            raise MyException(e, sys) from e
        
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        """
        Initiates the model evaluation process.
        """
        try:
            logging.info("Initiating model evaluation.")
            evaluation_artifact = self.evaluate_model()
            logging.info("Model evaluation process finished.")
            return evaluation_artifact
        except Exception as e:
            raise MyException(e,sys)
