# GitHub Actions and GCP Connections: Lab 5

## Overview
This lab demonstrates how to automate an end-to-end machine learning workflow using **GitHub Actions** and **Google Cloud Platform (GCP)**.  
You will create a CI pipeline that:

- Trains a machine learning model using Python  
- Authenticates securely to GCP using a service account  
- Uploads the trained model to a Google Cloud Storage (GCS) bucket  
- Runs manually or on a daily schedule  

For this lab, we use the **Breast Cancer dataset** and train a **Logistic Regression** classification model.

---

## Learning Objectives
By completing this lab, you will learn how to:

1. Configure a GCP project for automated ML workflows  
2. Create a service account and securely authenticate from GitHub  
3. Use GitHub Actions to automate ML model training  
4. Store trained ML artifacts (model files) inside a GCS bucket  
5. Schedule automated workflows using cron  

---

## Project Structure

```
lab_5/
├── train_and_save_model.py        # ML training + GCS upload script
├── requirements.txt                # Python dependencies
├── README.md                       # (this file)
└── .github/
    └── workflows/
        └── train-and-upload.yml    # GitHub Actions workflow
```

### `train_and_save_model.py`
This script:

- Loads the **Breast Cancer dataset**
- Builds a `Pipeline(StandardScaler() + LogisticRegression())`
- Splits data and trains the model
- Calculates accuracy
- Saves model as `model.joblib`
- Uploads the model to:

```
gs://mlops-lab5-models/trained_models/
```

---

## GCP Setup

### 1. Create/Select a GCP Project
Use the GCP Console and select or create a project.

### 2. Create a GCS Bucket
Navigate to:

**Cloud Storage → Buckets → Create**

Bucket used in this lab:

```
mlops-lab5-models
```

### 3. Create a Service Account
Go to:

**IAM & Admin → Service Accounts → Create Service Account**

- Name: `github-actions-sa`
- Role: **Storage Admin**

### 4. Generate JSON Key
Inside the service account:

**Keys → Add Key → Create new key (JSON)**

Download this file.

### 5. Add GitHub Secret
In your GitHub repo:

**Settings → Secrets and Variables → Actions**

Create a new secret:

- **Name:** `GCP_SA_KEY`  
- **Value:** *(paste the full JSON key)*

---

## GitHub Actions Workflow

The automation is configured in:

```
.github/workflows/train-and-upload.yml
```

This workflow:

- Runs nightly at midnight (cron schedule)
- Allows manual triggers (`workflow_dispatch`)
- Sets up Python 3.10
- Installs dependencies with caching
- Authenticates to GCP using the service account key
- Runs the ML training + upload step

Important environment variable:

```yaml
env:
  GCS_MODEL_BUCKET: mlops-l
