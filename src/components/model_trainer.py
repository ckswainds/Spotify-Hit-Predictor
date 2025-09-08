import os, sys
import numpy as np
import pickle, json, joblib
import optuna
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ClassificationMetricArtifact
)
from src.entity.estimator import MyModel


class ModelTrainer:
    """
    Handles Optuna hyperparameter tuning, MLflow experiment tracking,
    best model selection, and saving final model + metrics.
    """

    def __init__(self, model_trainer_config: ModelTrainerConfig, data_tranformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer Log Started {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_tranformation_artifact
            self.preprocessor_object_file_path = self.data_transformation_artifact.preprocessor_object_file_path

            logging.info("Loading transformed training and testing data.")
            train = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test = np.load(self.data_transformation_artifact.transformed_test_file_path)

            self.X_train, self.y_train = train["X"], train["y"].reshape(-1)
            self.X_test, self.y_test = test["X"], test["y"].reshape(-1)

            logging.info("Data loaded successfully.")

        except Exception as e:
            raise MyException(e, sys)

    def _objective(self, trial: optuna.trial) -> float:
        """
        Optuna objective function: train model with given hyperparams,
        log to MLflow, and return accuracy.
        """
        try:
            logging.info(f"Starting Optuna trial {trial.number}.")

            # Choose classifier
            classifier_name = trial.suggest_categorical("classifier_name", ["RandomForestClassifier", "XGBClassifier"])

            if classifier_name == "RandomForestClassifier":
                rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 200)
                rf_max_depth = trial.suggest_int("rf_max_depth", 2, 10, log=True)
                model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
            else:
                xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200)
                xgb_max_depth = trial.suggest_int("xgb_max_depth", 2, 10)
                model = XGBClassifier(
                    n_estimators=xgb_n_estimators,
                    max_depth=xgb_max_depth,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                )

            # Each Optuna trial = separate MLflow run
            with mlflow.start_run(nested=True):
                mlflow.log_param("classifier", classifier_name)
                mlflow.log_params(trial.params)

                model.fit(self.X_train, self.y_train)
                y_preds = model.predict(self.X_test)

                # Evaluate
                accuracy = accuracy_score(self.y_test, y_preds)
                precision = precision_score(self.y_test, y_preds)
                recall = recall_score(self.y_test, y_preds)
             

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                

                # Save trial model
                mlflow.sklearn.log_model(model, f"trial_{trial.number}_model")

            logging.info(f"Trial {trial.number} completed with accuracy: {accuracy}")
            return accuracy

        except Exception as e:
            logging.error(f"Error during Optuna trial {trial.number}.", exc_info=True)
            raise MyException(e, sys) from e

    def get_model_object_and_report(self) -> tuple[MyModel, ClassificationMetricArtifact]:
        """
        Run Optuna study, pick best model, re-train, and return final model + metrics.
        """
        try:
            logging.info("Starting Optuna + MLflow study.")

            mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns')}")
            mlflow.set_experiment("Spotify-Model-Tuning")

            with mlflow.start_run(run_name="Optuna_Study") as parent_run:
                study = optuna.create_study(direction="maximize")
                study.optimize(self._objective, n_trials=20)

                best_trial = study.best_trial
                best_model_params = best_trial.params
                best_model_name = best_model_params["classifier_name"]

                logging.info(f"Best trial {best_trial.number} | Accuracy: {best_trial.value}")

                # Log best trial summary
                mlflow.log_params(best_model_params)
                mlflow.log_metric("best_accuracy", best_trial.value)

            # Retrain best model
            if best_model_name == "RandomForestClassifier":
                best_model = RandomForestClassifier(
                    n_estimators=best_model_params["rf_n_estimators"],
                    max_depth=best_model_params["rf_max_depth"],
                    random_state=42
                )
            else:
                best_model = XGBClassifier(
                    n_estimators=best_model_params["xgb_n_estimators"],
                    max_depth=best_model_params["xgb_max_depth"],
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                )

            best_model.fit(self.X_train, self.y_train)
            y_preds = best_model.predict(self.X_test)

            # Preprocessor
            preprocessor = joblib.load(self.preprocessor_object_file_path)
            final_model = MyModel(preprocessing_object=preprocessor, trained_model_object=best_model)

            # Metrics
            metric_artifact = ClassificationMetricArtifact(
                accuracy_score=accuracy_score(self.y_test, y_preds),
                recall_score=recall_score(self.y_test, y_preds),
                precision_score=precision_score(self.y_test, y_preds),
            
            )

            # Save report JSON
            metrics_dict = {
                "Model_name": best_model_name,
                "accuracy_score": metric_artifact.accuracy_score,
                "recall_score": metric_artifact.recall_score,
                "precision_score": metric_artifact.precision_score,
              
            }

            report_dir = os.path.dirname(self.model_trainer_config.trained_model_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            with open(self.model_trainer_config.trained_model_report_file_path, 'w') as f:
                json.dump(metrics_dict, f, indent=4)

            return final_model, metric_artifact

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrates full training: runs Optuna, selects best model,
        saves MyModel + report, and returns artifact.
        """
        try:
            logging.info("Initiating model trainer.")

            # Get final model + metrics
            mymodel, metric_artifact = self.get_model_object_and_report()

            # Save final model
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)

            with open(self.model_trainer_config.trained_model_file_path, 'wb') as f:
                pickle.dump(mymodel, f)

            # Create artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_file_path,
                trained_model_scores=metric_artifact
            )

            logging.info(f"Final model saved at {self.model_trainer_config.trained_model_file_path}")
            logging.info(f"{'>>'*20} Model Trainer Log Completed {'<<'*20}")

            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)
