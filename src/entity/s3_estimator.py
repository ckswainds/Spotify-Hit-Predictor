import sys
from pandas import DataFrame
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel   

class Proj1Estimator:
    """
    Handles saving, loading, and predicting with model stored in S3.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        :param bucket_name: S3 bucket name
        :param model_path: Path to model inside bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: MyModel = None

    def is_model_present(self, model_path: str) -> bool:
        """
        Check if the model exists in S3.
        """
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name, s3_key=model_path
            )
        except MyException as e:
            print(e)
            return False

    def load_model(self) -> MyModel:
        """
        Load MyModel object from S3.
        """
        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Save model to S3.

        :param from_file: Local path of model file
        :param remove: If True, delete local file after upload
        """
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        Make predictions using loaded model.
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise MyException(e, sys)
