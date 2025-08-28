<<<<<<< HEAD
# Spotify Hit Predictor: An MLOps Project 🎶

This project applies **Machine Learning Operations (MLOps)** principles to predict whether a song will be a hit or a flop on Spotify based on its audio features. By building an end-to-end MLOps pipeline, I ensured that the model is **robust, scalable, and production-ready**.

---

## 🎯 Project Overview

The goal of this project was to build a machine learning model that can classify a song as either a **"hit"** or a **"flop"** by analyzing audio features such as loudness, liveness, danceability, and more.

I structured the entire project using **Object-Oriented Programming (OOP)** principles to maintain **modularity, reusability, and clarity**.

---

## ✨ MLOps Pipeline & Key Features

This project goes beyond a standalone machine learning model—it's designed as a **complete MLOps solution** for real-world deployment.

* **Data Handling & Processing**:
  I implemented modules for **data ingestion, validation, and transformation** to guarantee clean, reliable inputs for model training.

* **Model Training & Optimization**:

  * **MLflow**: Used for **experiment tracking** to log and compare parameters, metrics, and artifacts.
  * **Optuna**: Integrated for **hyperparameter tuning**, automatically searching for the best configurations.

* **CI/CD (Continuous Integration/Continuous Deployment)**:

  * **GitHub Actions** automates the pipeline so that any code push triggers testing, building, and deployment steps.

* **Cloud Infrastructure**:

  * **AWS S3**: Stores the trained models.
  * **AWS ECR**: Hosts Docker images of the application.
  * **AWS EC2**: Serves the application in a scalable cloud environment.

* **API Service**:

  * **FastAPI** powers a REST API that allows real-time predictions. Song features can be sent as JSON requests, and the API responds with predictions.

---

## 📂 Project Structure

```
.
├── artifacts/
│   ├── data_ingestion/
│   ├── data_transformation/
│   ├── data_validation/
│   ├── model_evaluation/
│   └── model_trainer/
├── notebooks/
│   ├── data/
│   └── (Jupyter notebooks for experimentation)
├── src/
│   ├── cloud_storage/
│   │   └── aws_storage.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_evaluation.py
│   │   ├── model_pusher.py
│   │   └── model_trainer.py
│   ├── configuration/
│   │   ├── aws_connection.py
│   │   └── mongo_db_connection.py
│   ├── constants/
│   ├── data_access/
│   │   └── proj1_data.py
│   ├── entity/
│   │   ├── artifact_entity.py
│   │   ├── config_entity.py
│   │   ├── estimator.py
│   │   └── s3_estimator.py
│   ├── exception/
│   ├── logger/
│   ├── pipeline/
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   └── utils/
│       └── main_utils.py
├── static/
│   └── css/
│       └── style.css
├── templates/
│   └── index.html
├── .github/
│   └── workflows/
│       └── ci_cd.yaml
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

* **Python**: Core programming language.
* **Scikit-learn, XGBoost**: Machine learning libraries.
* **MLflow**: For tracking experiments.
* **Optuna**: Hyperparameter optimization.
* **FastAPI**: Web framework for serving the model.
* **Docker**: Containerization.
* **GitHub Actions**: CI/CD automation.
* **AWS**:

  * **S3**: Model storage.
  * **ECR**: Docker image registry.
  * **EC2**: Cloud deployment.

---

## ⚙️ How to Run the Project

### Prerequisites

* Python 3.8+
* Docker
* AWS account with configured credentials
* GitHub account

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/ckswainds/Spotify_tracks_classification.git
   cd spotify-hit-predictor
   ```

2. **Set up AWS Environment**:

   * Create an S3 bucket for storing models.
   * Create an ECR repository for Docker images.
   * Configure AWS credentials.

3. **Local Setup**:

   * Install dependencies:

     ```bash
     pip install -r requirements.txt
     ```
   * Run the training pipeline:

     ```bash
     python main.py
     ```

4. **Deployment (via CI/CD)**:

   * GitHub Actions will automatically:

     1. Build the Docker image.
     2. Push it to AWS ECR.
     3. Deploy it on AWS EC2.

5. **Access the API**:

   * The FastAPI service runs at:
     `http://<EC2_PUBLIC_IP>:8000/predict`
   * Send a `POST` request with song features (JSON) to receive predictions.

---

## 🤝 Contribution

=======
>>>>>>> 4f068a5 (DONE)
