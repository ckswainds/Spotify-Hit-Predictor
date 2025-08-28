import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    """
    Maps your project labels: 
    hit -> 1
    flop -> 0
    """
    def __init__(self):
        self.hit: int = 1
        self.flop: int = 0

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class MyModel:
    """
    Wrapper around preprocessing pipeline + trained ML model.
    Ensures preprocessing and prediction are applied consistently.
    """

    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Preprocessing pipeline (scaler, encoder, etc.)
        :param trained_model_object: Trained ML model (sklearn, xgboost, etc.)
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Applies preprocessing + prediction.
        """
        try:
            logging.info("Starting prediction process.")

            # Step 1: Apply preprocessing (scaling, encoding, etc.)
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Step 2: Predict using trained model
            logging.info("Using trained model to get predictions.")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
