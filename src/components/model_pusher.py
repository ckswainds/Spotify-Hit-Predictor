import sys
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact,ModelEvaluationArtifact,ModelTrainerArtifact
from src.entity.s3_estimator import Proj1Estimator


class ModelPusher:
    """
    This class is responsible for pushing a newly trained model to the
    production environment (S3 bucket) if it has been accepted
    by the model evaluation stage.
    """
    def __init__(self,model_pusher_config:ModelPusherConfig,
                 model_evaluation_artifact:ModelEvaluationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        """
        Initializes the ModelPusher class.

        Args:
            model_pusher_config (ModelPusherConfig): Configuration for the model pusher.
            model_evaluation_artifact (ModelEvaluationArtifact): The artifact from the model evaluation stage.
            model_trainer_artifact (ModelTrainerArtifact): The artifact from the model training stage.
        """
        try:
            logging.info(f"{'>>'*20} Model Pusher Log Started {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            
            logging.info("Creating Proj1Estimator instance for S3 operations.")
            self.proj1_estimator = Proj1Estimator(
                bucket_name=model_pusher_config.bucket_name,
                model_path=self.model_pusher_config.s3_model_key_path
            )
            logging.info("ModelPusher initialized successfully.")
            
        except Exception as e:
            raise MyException(e,sys)
        
    
    def initiate_model_pusher(self)->ModelPusherArtifact:
        """
        Initiates the model pushing process. The model will only be pushed
        if the evaluation artifact indicates it is accepted.

        Returns:
            ModelPusherArtifact: An artifact containing the S3 bucket and model key path.
        """
        try:
            logging.info("Starting model pushing process.")
            logging.info(f"Is the new model accepted? {self.model_evaluation_artifact.is_model_accepted}")
            
            if self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model accepted. Pushing model to S3 bucket.")
                
                # Pushing the trained model to the S3 bucket
                self.proj1_estimator.save_model(
                    from_file=self.model_trainer_artifact.trained_model_path
                )
                
                logging.info(f"Model successfully pushed to S3 at: {self.model_pusher_config.s3_model_key_path}")
                
                model_pusher_artifact = ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=self.model_pusher_config.s3_model_key_path
                )
                
                logging.info(f"Model Pusher Artifact: {model_pusher_artifact}")
                logging.info(f"{'>>'*20} Model Pusher Log Completed {'<<'*20}")
                return model_pusher_artifact
            
            else:
                logging.info("Model not accepted based on evaluation. Skipping model push.")
                model_pusher_artifact = ModelPusherArtifact(
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_model_path=None
                )
                logging.info("Model Pusher process completed without pushing.")
                return model_pusher_artifact
        
        except Exception as e:
            logging.error("An error occurred during model pushing.")
            raise MyException(e,sys)
