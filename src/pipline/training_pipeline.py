import sys
from src.exception import MyException 
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,
                                      DataTransformationConfig,
                                      ModelTrainerConfig,
                                      ModelEvaluationConfig,
                                      ModelPusherConfig
                                      )
from src.entity.artifact_entity import (DataIngestionArtifact,
                                        DataValidationArtifact,
                                        DataTransformationArtifact,
                                        ModelTrainerArtifact,
                                        ModelEvaluationArtifact,
                                        ModelPusherArtifact)

class TrainPipeline:
    """
    TrainPipeline Class:
    --------------------
    Orchestrates the entire training pipeline. 
    Currently, it includes only the data ingestion stage but 
    can be extended to include data validation, transformation, 
    model training, evaluation, and deployment.
    """

    def __init__(self):
        """
        Initialize the TrainPipeline with DataIngestionConfig.
        """
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config=DataValidationConfig()
            self.data_transformation_config=DataTransformationConfig()
            self.model_trainer_config=ModelTrainerConfig()
            self.model_evaluation_config=ModelEvaluationConfig()
            self.model_pusher_config=ModelPusherConfig()
            logging.info("TrainPipeline initialized with DataIngestionConfig")
        except Exception as e:
            logging.error("Error occurred during TrainPipeline initialization")
            raise MyException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Start the data ingestion process.
        
        Returns:
            DataIngestionArtifact: Contains file paths of train and test datasets.
        """
        logging.info("Entered start_data_ingestion method of TrainPipeline class")
        try:
            data_ingestion = DataIngestion(self.data_ingestion_config)
            logging.info("DataIngestion instance created successfully")

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact
        except Exception as e:
            logging.error("Error occurred in start_data_ingestion")
            raise MyException(e, sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        """
        Start the data validation process.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact from the data ingestion stage.

        Returns:
            DataValidationArtifact: The artifact from the data validation stage.
        """
        logging.info("Entered start_data_validation method of TrainPipeline class")
        try:
            logging.info("Creating DataValidation instance.")
            data_validation=DataValidation(self.data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
            
            logging.info("Initiating data validation.")
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info(f"Data validation completed. Artifact: {data_validation_artifact}")
            
            return data_validation_artifact
        except Exception as e:
            logging.error("Error occurred in start_data_validation")
            raise MyException(e,sys)
        
    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact,
                                  data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        """
        Start the data transformation process.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact from the data ingestion stage.
            data_validation_artifact (DataValidationArtifact): The artifact from the data validation stage.

        Returns:
            DataTransformationArtifact: The artifact from the data transformation stage.
        """
        logging.info("Entered start_data_transformation method of TrainPipeline class")
        try:
            logging.info("Creating DataTransformation instance.")
            data_transformation=DataTransformation(self.data_transformation_config,
                                                   data_ingestion_artifact=data_ingestion_artifact,
                                                   data_validation_artifact=data_validation_artifact)
            
            logging.info("Initiating data transformation.")
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            
            return data_transformation_artifact
        except Exception as e:
            logging.error("Error occurred in start_data_transformation")
            raise MyException(e, sys)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        """
        Starts the model training process.

        Args:
            data_transformation_artifact (DataTransformationArtifact): The artifact from the data transformation stage.

        Returns:
            ModelTrainerArtifact: The artifact from the model trainer stage, containing the trained model and its metrics.
        """
        logging.info("Entered start_model_trainer method of TrainPipeline class")
        try:
            logging.info("Creating ModelTrainer instance.")
            model_trainer=ModelTrainer(self.model_trainer_config,data_transformation_artifact)
            
            logging.info("Initiating model training.")
            model_trainer_artifact=model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed. Artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
        except Exception as e:
            logging.error("Error occurred in start_model_trainer")
            raise MyException(e, sys)
        
    def start_model_evaluation(self,model_trainer_artifact:ModelTrainerArtifact,data_transformation_artifact:DataTransformationArtifact,data_ingestion_artifact:DataIngestionArtifact)->ModelEvaluationArtifact:
        """
        Starts the model evaluation process to compare the new model with the production model.

        Args:
            model_trainer_artifact (ModelTrainerArtifact): The artifact from the model training stage.
            data_transformation_artifact (DataTransformationArtifact): The artifact from the data transformation stage.

        Returns:
            ModelEvaluationArtifact: An artifact containing the evaluation results.
        """
        logging.info("Entered start_model_evaluation method of TrainPipeline class")
        try:
            logging.info("Creating ModelEvaluation instance.")
            model_evaluation=ModelEvaluation(model_evaluation_config=self.model_evaluation_config,
                                             model_trainer_artifact=model_trainer_artifact,
                                             data_transformation_artifact=data_transformation_artifact,
                                             data_ingestion_artifact=data_ingestion_artifact)
            
            logging.info("Initiating model evaluation.")
            model_evaluation_artifact=model_evaluation.initiate_model_evaluation()
            logging.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")

            return model_evaluation_artifact
        except Exception as e:
            logging.error("Error occurred in start_model_evaluation")
            raise MyException(e,sys)
        
    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact,model_trainer_artifact:ModelTrainerArtifact)->ModelPusherArtifact:
        """
        Starts the model pushing process to move the new model to production if accepted.

        Args:
            model_evaluation_artifact (ModelEvaluationArtifact): The artifact from the model evaluation stage.
            model_trainer_artifact (ModelTrainerArtifact): The artifact from the model training stage.

        Returns:
            ModelPusherArtifact: An artifact containing the result of the push operation.
        """
        logging.info("Entered start_model_pusher method of TrainPipeline class")
        try:
            logging.info("Creating ModelPusher instance.")
            model_pusher=ModelPusher(model_pusher_config=self.model_pusher_config,
                                     model_evaluation_artifact=model_evaluation_artifact,
                                     model_trainer_artifact=model_trainer_artifact)
            
            logging.info("Initiating model pushing.")
            model_pusher_artifact=model_pusher.initiate_model_pusher()
            logging.info(f"Model pushing completed. Artifact: {model_pusher_artifact}")

            return model_pusher_artifact
        except Exception as e:
            logging.error("Error occurred in start_model_pusher")
            raise MyException(e,sys)
        

    def run_pipeline(self) -> None:
        """
        Run the entire machine learning pipeline.
        This method orchestrates the sequential execution of each stage:
        - Data Ingestion
        - Data Validation
        - Data Transformation
        - Model Training
        - Model Evaluation
        - Model Pusher
        
        Returns:
            None
        """
        logging.info("Pipeline execution started")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data Ingestion stage completed successfully")
            
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            logging.info("Data Validation stage completed successfully")
            
            data_transformation_artifact=self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                        data_validation_artifact=data_validation_artifact)
            logging.info("Data Transformation stage completed successfully")
            
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info("Model Training stage completed successfully")
            
            model_evaluation_artifact=self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,
                                                                  data_transformation_artifact=data_transformation_artifact,
                                                                  data_ingestion_artifact=data_ingestion_artifact
                                                                  )
            logging.info("Model Evaluation stage completed successfully")
            
            model_pusher_artifact=self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact,
                                                          model_trainer_artifact=model_trainer_artifact)
            logging.info("Model Pusher stage completed successfully")
            
            logging.info("Pipeline execution finished")

        except Exception as e:
            logging.error("Error occurred during pipeline execution")
            raise MyException(e, sys)
