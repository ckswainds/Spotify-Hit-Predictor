# from src.entity.config_entity import SpotifyHitPredictorConfig
# from src.exception import MyException
# from src.logger import logging
# import numpy as np
# from src.entity.s3_estimator import Proj1Estimator
# import sys
# from pandas import DataFrame
# import joblib

# import pandas as pd
# from typing import Optional


# class SpotifyData:
#     """
#     A data class to hold a single record of Spotify audio features and track information.
#     This class is designed to structure raw input data for a machine learning model.
#     """
#     def __init__(self,
#                  danceability: float,
#                  energy: float,
#                  key: int,
#                  loudness: float,
#                  mode: int,
#                  speechiness: float,
#                  acousticness: float,
#                  instrumentalness: float,
#                  liveness: float,
#                  valence: float,
#                  tempo: float,
#                  duration_ms: int,
#                  time_signature: int,
#                  chorus_hit: float,
#                  sections: int,
#                  ):
#         """
#         Initializes a SpotifyData object with song features.

#         Args:
#             danceability (float): Describes how suitable a track is for dancing.
#             energy (float): Perceptual measure of intensity and activity.
#             key (int): The key the track is in (e.g., 0-11).
#             loudness (float): The overall loudness of the track in decibels (dB).
#             mode (int): The modality of a track (major or minor).
#             speechiness (float): The presence of spoken words in a track.
#             acousticness (float): A confidence measure of whether the track is acoustic.
#             instrumentalness (float): The probability that the track contains no vocals.
#             liveness (float): Detects the presence of an audience in the recording.
#             valence (float): Describes the musical positiveness conveyed by a track.
#             tempo (float): The overall estimated tempo of a track in beats per minute (BPM).
#             duration_ms (int): The duration of the track in milliseconds.
#             time_signature (int): The number of beats per measure.
#             chorus_hit (float): A measure of how much of the chorus is hit.
#             sections (int): The number of sections in the track.
#             target (Optional[int]): The target variable (e.g., popularity, classification label).
#                                    Defaults to None, as it's optional for new data prediction.
#         """
#         try:
            
#             self.danceability = danceability
#             self.energy = energy
#             self.key = key
#             self.loudness = loudness
#             self.mode = mode
#             self.speechiness = speechiness
#             self.acousticness = acousticness
#             self.instrumentalness = instrumentalness
#             self.liveness = liveness
#             self.valence = valence
#             self.tempo = tempo
#             self.duration_ms = duration_ms
#             self.time_signature = time_signature
#             self.chorus_hit = chorus_hit
#             self.sections = sections
          
            
#         except Exception as e:
#             raise MyException(e, sys) from e

#     def get_data_as_dict(self) -> dict:
#         """Converts the object's attributes into a dictionary."""
#         return self.__dict__
    
#     def get_data_as_dataframe(self) -> pd.DataFrame:
#         """
#         Converts the SpotifyData object into a pandas DataFrame.
#         This format is ideal for model prediction.
#         """
#         try:
#             logging.info("Converting SpotifyData object to a DataFrame.")
#             data_dict = self.get_data_as_dict()
#             df = pd.DataFrame([data_dict])
#             logging.info("DataFrame created successfully.")
#             return df
#         except Exception as e:
#             logging.error("Failed to convert data to DataFrame.", exc_info=True)
#             raise MyException(e, sys) from e

# class SpotifyHitPredictor:
#     def __init__(self,spotify_hit_predictor_config:SpotifyHitPredictorConfig):
#         try:
#            self.spotify_hit_predictor_config=spotify_hit_predictor_config
#         except Exception as e:
#             raise MyException(e, sys)
#     def predict(self,df:DataFrame)->str:
#         try:
#             # model=Proj1Estimator(
#             #     bucket_name=self.spotify_hit_predictor_config.model_bucket_name,
#             #     model_path=self.spotify_hit_predictor_config.model_file_name
#             # )
#             model=joblib.load("C:\Vscode\git\mlops\Spotify_tracks_classification\artifacts\08_21_2025_20_40_31\model_trainer\model.pkl")
#             result=model.predict(dataframe=df)
#             return result
        
#         except Exception as e:
#             raise MyException(e, sys)


import sys
import pandas as pd
from src.exception import MyException
from src.logger import logging
from src.entity.s3_estimator import Proj1Estimator  # Assuming this loads your model
# from src.entity.spotify_data import SpotifyData 
from src.entity.config_entity import SpotifyHitPredictorConfig
import joblib
# Your data class
import sys
import pandas as pd
from typing import Optional
from src.exception import MyException
from src.logger import logging

class SpotifyData:
    """
    A data class to hold a single record of Spotify audio features and track information.
    This class is designed to structure raw input data for a machine learning model.
    """
    def __init__(self,
                
                 danceability: float,
                 energy: float,
                 key: int,
                 loudness: float,
                 mode: int,
                 speechiness: float,
                 acousticness: float,
                 instrumentalness: float,
                 liveness: float,
                 valence: float,
                 tempo: float,
                 duration_ms: int,
                 time_signature: int,
                 chorus_hit: float,
                 sections: int,
                 
            
                 ):
        """
        Initializes a SpotifyData object with song features.

        Args:
            track (str): The name of the track.
            artist (str): The name of the artist.
            uri (str): The Spotify URI for the track.
            danceability (float): Describes how suitable a track is for dancing.
            energy (float): Perceptual measure of intensity and activity.
            key (int): The key the track is in (e.g., 0-11).
            loudness (float): The overall loudness of the track in decibels (dB).
            mode (int): The modality of a track (major or minor).
            speechiness (float): The presence of spoken words in a track.
            acousticness (float): A confidence measure of whether the track is acoustic.
            instrumentalness (float): The probability that the track contains no vocals.
            liveness (float): Detects the presence of an audience in the recording.
            valence (float): Describes the musical positiveness conveyed by a track.
            tempo (float): The overall estimated tempo of a track in beats per minute (BPM).
            duration_ms (int): The duration of the track in milliseconds.
            time_signature (int): The number of beats per measure.
            chorus_hit (float): A measure of how much of the chorus is hit.
            sections (int): The number of sections in the track.
            target (Optional[int]): The target variable (e.g., a classification label).
                                   Defaults to None, as it's optional for new data prediction.
        """
        try:
            self.danceability = danceability
            self.energy = energy
            self.key = key
            self.loudness = loudness
            self.mode = mode
            self.speechiness = speechiness
            self.acousticness = acousticness
            self.instrumentalness = instrumentalness
            self.liveness = liveness
            self.valence = valence
            self.tempo = tempo
            self.duration_ms = duration_ms
            self.time_signature = time_signature
            self.chorus_hit = chorus_hit
            self.sections = sections
          
            
        except Exception as e:
            raise MyException(e, sys) from e

    def get_data_as_dict(self) -> dict:
        """
        Converts the object's attributes into a dictionary.
        This function is useful for creating a single-row DataFrame.
        """
        logging.info("Converting SpotifyData object to a dictionary.")
        try:
            input_data = {
               
                "danceability": [self.danceability],
                "energy": [self.energy],
                "key": [self.key],
                "loudness": [self.loudness],
                "mode": [self.mode],
                "speechiness": [self.speechiness],
                "acousticness": [self.acousticness],
                "instrumentalness": [self.instrumentalness],
                "liveness": [self.liveness],
                "valence": [self.valence],
                "tempo": [self.tempo],
                "duration_ms": [self.duration_ms],
                "time_signature": [self.time_signature],
                "chorus_hit": [self.chorus_hit],
                "sections": [self.sections],
                
            }
            logging.info("Successfully created dictionary.")
            return input_data
        except Exception as e:
            logging.error("Failed to convert data to dictionary.", exc_info=True)
            raise MyException(e, sys) from e
    
    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Converts the SpotifyData object into a pandas DataFrame.
        This format is ideal for model prediction.
        """
        try:
            logging.info("Converting SpotifyData object to a DataFrame.")
            data_dict = self.get_data_as_dict()
            df = pd.DataFrame(data_dict)
            logging.info("DataFrame created successfully.")
            return df
        except Exception as e:
            logging.error("Failed to convert data to DataFrame.", exc_info=True)
            raise MyException(e, sys) from e


class PredictionPipeline:
    model = None   # class-level variable (shared by all instances)

    def __init__(self):
        """
        Initializes the prediction pipeline by loading the trained model
        only once per session.
        """
        try:
            if PredictionPipeline.model is None:   # load model only first time
                spotify_prediction_config = SpotifyHitPredictorConfig()
                PredictionPipeline.model = Proj1Estimator(
                    bucket_name=spotify_prediction_config.model_bucket_name,
                    model_path=spotify_prediction_config.model_file_name
                )
            # PredictionPipeline.model=joblib.load("C:/Vscode/git/mlops/Spotify_tracks_classification/artifacts/08_22_2025_20_26_18/model_trainer/model.pkl")

            # now all instances can access this model without reloading
            self.model = PredictionPipeline.model

        except Exception as e:
            logging.error("Failed to initialize PredictionPipeline.")
            raise MyException(e, sys)

    def predict(self, data: SpotifyData):
        """
        Takes a SpotifyData object, preprocesses it, and returns a prediction.
        """
        try:
            # 2. Convert the input data to a DataFrame
            df = data.get_data_as_dataframe()

            # # 3. Drop non-predictive columns (if any)
            # # Example: df.drop(columns=['track', 'artist'], inplace=True)
            
            # # 4. Preprocess the data using the loaded preprocessor
            # transformed_data = self.preprocessor.transform(df)

            # 5. Get the prediction from the trained model
            prediction = self.model.predict(df)

            # 6. Map the prediction to a human-readable label
            # Assuming 0 is 'Hit' and 1 is 'Not a Hit'
            status =prediction[0]
            # print(status)
            return status

        except Exception as e:
            logging.error("Prediction failed.")
            raise MyException(e, sys)