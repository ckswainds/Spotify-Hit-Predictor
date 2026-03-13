
# Spotify Hit Predictor: Production-Grade MLOps Pipeline

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Containerized-brightblue)](Dockerfile)
[![AWS](https://img.shields.io/badge/Cloud-AWS-orange)](https://aws.amazon.com/)

## Executive Summary

A **production-grade MLOps pipeline** that predicts Spotify hit songs with automated model deployment, experiment tracking, and hyperparameter optimization. This end-to-end project demonstrates enterprise-level ML engineering practices including modular architecture, CI/CD automation, and cloud infrastructure management.

**Key Highlights:**
- ✅ **Automated ML Pipeline**: Data ingestion → validation → transformation → model training → deployment
- ✅ **Experiment Tracking**: MLflow integration for reproducible, comparable models
- ✅ **Hyperparameter Optimization**: Optuna-powered automated tuning for optimal performance
- ✅ **Production Deployment**: Docker containerization with AWS infrastructure (ECR, S3, EC2)
- ✅ **REST API**: FastAPI-powered real-time prediction endpoint
- ✅ **CI/CD Automation**: GitHub Actions for automated testing and deployment

---

## Project Vision

This project demonstrates the complete ML lifecycle beyond model development—building a **scalable, maintainable, and deployable machine learning system**. The application predicts whether a song will be a hit or flop on Spotify by analyzing audio features (loudness, danceability, energy, etc.), with a focus on production-ready code and enterprise patterns.

---

## Technical Architecture

### Pipeline Components

#### 1. **Data Layer**
- **Data Ingestion**: Automated data collection and import from multiple sources
- **Data Validation**: Schema validation and quality checks to ensure data integrity
- **Data Transformation**: Feature engineering, normalization, and preprocessing for model consumption

#### 2. **ML Training & Optimization**
- **Model Training**: Scikit-learn and XGBoost implementations with cross-validation
- **Hyperparameter Tuning**: Optuna framework for automated, intelligent parameter optimization
- **Experiment Tracking**: MLflow integration for logging metrics, parameters, and model artifacts across training runs
- **Model Evaluation**: Comprehensive validation using multiple metrics and performance benchmarks

#### 3. **Model Persistence & Serving**
- **Model Registry**: Automated versioning and management of trained models
- **Cloud Storage**: AWS S3 for scalable, persistent model artifact storage
- **REST API**: FastAPI endpoints for real-time predictions with JSON request/response handling

#### 4. **Infrastructure & Deployment**
- **Containerization**: Docker for consistent, reproducible environments across development, testing, and production
- **Container Registry**: AWS ECR for centralized Docker image hosting
- **Cloud Compute**: AWS EC2 for scalable, on-demand application hosting
- **CI/CD Pipeline**: GitHub Actions for automated workflows:
  - ✓ Code testing on push
  - ✓ Docker image building
  - ✓ ECR image push
  - ✓ EC2 deployment automation

### Code Quality & Architecture

- **Object-Oriented Design**: Modular, reusable components following SOLID principles
- **Configuration Management**: YAML-based configuration for reproducible training runs
- **Exception Handling**: Custom exception hierarchy for robust error management
- **Logging**: Structured logging throughout the pipeline for debugging and monitoring
- **Utils & Helpers**: Centralized utility functions for common operations

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

## Tech Stack

| Category | Technologies |
|----------|---------------|
| **Language** | Python 3.8+ |
| **ML Frameworks** | Scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow |
| **Hyperparameter Optimization** | Optuna |
| **Web Framework** | FastAPI |
| **Containerization** | Docker |
| **Cloud Services** | AWS (S3, ECR, EC2) |
| **CI/CD** | GitHub Actions |
| **Database** | MongoDB |

---

## Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.8+ |
| Docker | Latest |
| AWS CLI | Configured with credentials |
| Git | Latest |

### Installation & Setup

#### Step 1: Clone the Repository

```bash
git clone https://github.com/ckswainds/Spotify-Hit-Predictor.git
cd Spotify-Hit-Predictor
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID and Secret Access Key
```

Create S3 bucket and ECR repository:
- S3 Bucket: For model artifact storage
- ECR Repository: For Docker image hosting

#### Step 5: Run Training Pipeline

```bash
python main.py
```

This command executes the complete pipeline:
1. Data ingestion from source
2. Data validation and quality checks
3. Feature transformation and preprocessing
4. Model training with hyperparameter optimization
5. Model evaluation and metrics logging
6. Model artifact storage in S3

### Local Testing

Test the prediction API locally:

```bash
python app.py
# API runs at http://localhost:8000
```

**Make a prediction request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 0.7, 0.8, 0.6, ...]
  }'
```

### Docker Deployment

Build and run locally:

```bash
docker build -t spotify-predictor .
docker run -p 8000:8000 spotify-predictor
```

### Production Deployment (AWS)

The CI/CD pipeline automates deployment:

1. **Trigger**: Push code to main branch
2. **Build**: GitHub Actions builds Docker image
3. **Push**: Image pushed to AWS ECR
4. **Deploy**: Automatically deployed to AWS EC2
5. **Access**: API available at `http://<EC2_PUBLIC_IP>:8000`

Configure CI/CD in `.github/workflows/ci_cd.yaml`

---

## Key Learnings & Skills Demonstrated

### MLOps & DevOps
- End-to-end ML pipeline design and implementation
- Experiment tracking and reproducibility (MLflow)
- Automated model tuning and optimization (Optuna)
- Containerization and orchestration (Docker)
- CI/CD pipeline automation (GitHub Actions)

### Cloud & Infrastructure
- AWS cloud services (S3, ECR, EC2)
- Infrastructure as Code principles
- Scalable deployment strategies
- Model versioning and registry management

### Software Engineering
- Object-oriented design patterns
- Modular, maintainable code architecture
- Configuration management best practices
- Exception handling and logging
- API design and RESTful principles

### Data & ML
- End-to-end feature engineering
- Model evaluation and validation
- Hyperparameter optimization strategies
- Cross-validation and performance metrics

---

## Project Structure

```
spotify-hit-predictor/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py          # Data collection & import
│   │   ├── data_validation.py         # Schema & quality validation
│   │   ├── data_transformation.py     # Feature engineering
│   │   ├── model_trainer.py           # Model training & tuning
│   │   ├── model_evaluation.py        # Performance evaluation
│   │   └── model_pusher.py            # Model registry & S3 upload
│   ├── configuration/
│   │   ├── aws_connection.py          # AWS client setup
│   │   └── mongo_db_connection.py     # Database connection
│   ├── pipeline/
│   │   ├── training_pipeline.py       # Orchestrates training flow
│   │   └── prediction_pipeline.py     # Orchestrates inference flow
│   ├── entity/                        # Data models & entities
│   ├── exception/                     # Custom exception classes
│   ├── logger/                        # Logging utilities
│   ├── data_access/                   # Database access layer
│   ├── cloud_storage/                 # AWS integrations
│   └── utils/                         # Helper functions
├── notebooks/
│   ├── spotify.ipynb                  # Exploratory analysis
│   └── data/                          # Notebook datasets
├── .github/
│   └── workflows/
│       └── ci_cd.yaml                 # GitHub Actions automation
├── config/
│   ├── schema.yaml                    # Data validation schema
│   └── model.yaml                     # Model configuration
├── artifacts/                         # Pipeline outputs (timestamped)
├── app.py                            # FastAPI application
├── main.py                           # Pipeline entry point
├── Dockerfile                        # Container definition
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Monitoring & Observability

- **MLflow Dashboard**: Track experiments, metrics, and model performance across runs
- **Logging**: Structured logs at each pipeline stage for debugging and monitoring
- **Error Tracking**: Custom exceptions with detailed error context
- **Model Artifacts**: Versioned models stored in S3 with metadata tracking

---

## Future Enhancements

- [ ] Model performance monitoring and alerting
- [ ] A/B testing capability for model comparisons
- [ ] Real-time data drift detection
- [ ] Advanced feature engineering with feature store
- [ ] Multi-model ensemble strategies
- [ ] API rate limiting and authentication

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Contact & Questions

For questions about the project architecture, implementation, or deployment strategy, feel free to reach out.

**Repository**: [GitHub - Spotify Hit Predictor](https://github.com/ckswainds/Spotify-Hit-Predictor)

---

*Built with ❤️ by [Your Name] | Last Updated: March 2026*
