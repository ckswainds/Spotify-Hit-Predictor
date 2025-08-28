import os

#Mongodb constants
DATABASE_NAME="spotify"
COLLECTION_NAME:str="spotifydata"
MONGODB_URL_KEY="MONGODB_URL"


MODEL_NAME="model.pkl"
DATA_FILE_NAME="dataset.csv"
ARTIFACT_DIR="artifacts"
DATA_FILE_TEMP_PATH="C:/Vscode/git/mlops/Spotify_tracks_classification/notebooks/data/spotify.csv"
SCHEMA_FILE_PATH=os.path.join("config", "schema.yaml")
PREPROCESSOR_OBJECT_FILE="preprocessing_object.pkl"

#AWS contsants
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "my-spotifymodel"
MODEL_PUSHER_S3_KEY = "model-registry"

#Data ingestion Constants
DATA_INGESTION_DIR_NAME="Data_ingestion"
DATA_INGESTION_RAW_DATA_FILE="spotify.csv"
DATA_INGESTION_INGESTED_DIR_NAME="ingested"
TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"
TRAIN_TEST_SPLIT_RATIO:float=0.30


#Data validation constants
DATA_VALIDATION_DIR:str="data_validation"
DATA_VALIDATION_REPORT_FILE_NAME="report.json"

#Data transformation constants
DATA_TRANSFORMATION_DIR:str='data_transformation'
TRANSFORMED_OBJECT_DIR:str="object"
TRANSFORMED_DATA_DIR:str="transformed_data"
TRANSFORMED_TRAIN_FILE:str="transformed_train_data.npz"
TRANSFORMED_TEST_FILE:str="transformed_test_data.npz"
# TRAIN_LABEL_FILE:str="train_label.npy"
# TEST_LABEL_FILE:str="test_label.npy"

# Model Trainer Constants
MODEL_TRAINER_DIR:str="model_trainer"
TRAINED_MODEL_REPORT_DIR:str="reports"
TRAINED_MODEL_REPORT_FILE:str="model_report.json"


APP_HOST = "0.0.0.0"
APP_PORT = 5000