# AI-Driven Pneumonia Detection Lab

This repository hosts an intelligent, automated system capable of detecting pneumonia from standard chest X-ray images and medical DICOM (`*.dcm`) files using deep learning (CNN) models.

## Project Structure
```text
pneumonia_detection/
├── deployment/
│   └── pneumonia_detection_model_v1_0.joblib   # Trained serialization pipeline (CNN 1)
└── pneumonia_detection_codefiles/
    ├── app_standalone.py                           # Standalone Streamlit application (Inline Inference)
    ├── Dockerfile                              # Multi-stage container instruction file
    └── requirements.txt                        # Strict library dependency versions
```

## Installation & Setup

### Local Python Environment
Ensure you have Python 3.10+ installed. Install the strictly-pinned dependencies locally using `pip`:

```bash
pip install -r pneumonia_detection/pneumonia_detection_codefiles/requirements.txt
```

### Standalone Application Run
To test or run the standalone UI locally without the Flask middleware proxy:

```bash
streamlit run pneumonia_detection/pneumonia_detection_codefiles/app_direct.py
```

### The app hosts an inhouse model to run the analysis of the uploaded image
 

## Containerization & Dockerization

A production-ready Dockerfile is prepared inside the project codebase to enable zero-configuration deployments on serverless cloud platforms such as AWS, GCP, or Hugging Face Spaces.

### Build the Docker Image
Navigate to the directory containing the application code and Dockerfile:

```bash
cd pneumonia_detection/pneumonia_detection_codefiles/
docker build -t pneumonia-detection-app:latest .
```

### Run the Container Locally
Expose the Streamlit web server outside of the docker network bridge:

```bash
docker run -d -p 8501:8501 pneumonia-detection-app:latest
```
Access the web application at `http://localhost:8501`.
