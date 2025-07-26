# Production-Ready MLOps Pipeline

[![CI/CD Pipeline](https://github.com/cm-upadhyay/mlops-starter-kit/actions/workflows/ci.yaml/badge.svg)](https://github.com/cm-upadhyay/mlops-starter-kit/actions)

This repository demonstrates a complete, end-to-end MLOps pipeline for training, testing, and deploying a machine learning model. The project automates the entire lifecycle, from a code change on a feature branch to a verified deployment in a production environment, following professional engineering practices.

## Key Features

-   **Automated Model Training & Testing**: A scikit-learn model is automatically validated on every pull request.
-   **CI/CD with GitHub Actions**: The entire workflow is orchestrated using GitHub Actions for robust automation.
-   **Containerization**: The model is served via a FastAPI application, containerized with Docker.
-   **Multi-Environment Deployment**: The pipeline automatically deploys to separate **Staging** and **Production** environments on Google Kubernetes Engine (GKE).
-   **Professional Git Workflow**: Utilizes a GitFlow-style branching strategy (`feature` -> `dev` -> `main`) to ensure code quality and stability.
-   **Automated Reporting**: CML (Continuous Machine Learning) is used to post test results and model metrics directly on pull requests.

## Technology Stack

-   **Model**: Python, Scikit-learn, Pandas
-   **API**: FastAPI, Uvicorn
-   **CI/CD**: GitHub Actions, CML
-   **Cloud & Deployment**: Google Cloud Platform (GCP), Docker, Kubernetes (GKE), Google Artifact Registry

## Development Environment

This project was developed and managed from a **Google Cloud Vertex AI Workbench** instance. This provided a unified JupyterLab environment with pre-installed data science libraries and seamless `gcloud` and `git` integration.

While development was done in Vertex AI, the setup and execution steps can be performed from any local machine with the required prerequisites installed.

## CI/CD Workflow

The pipeline follows a multi-stage process to safely move code from development to production:

1.  **`Feature Branch`**: All new work is done on a feature branch.
2.  **Pull Request to `dev`**: A PR triggers the `build-and-test` job.
    -   **CI Checks Pass**: Tests are run, the model is trained, and a CML report is posted.
3.  **Merge to `dev`**: The merge triggers a `push` event.
    -   **CD to Staging**: The application is automatically deployed to the staging GKE cluster.
4.  **Pull Request to `main`**: After staging is verified, a PR is created from `dev` to `main`.
    -   **CI Checks Pass**: The same validation job runs again as a final quality gate.
5.  **Merge to `main`**: The merge triggers the final `push` event.
    -   **CD to Production**: The application is automatically deployed to the production GKE cluster.

## Project Structure
.
├── .github
│   └── workflows
│       └── ci.yaml
├── data
│   └── iris.csv
├── k8s
│   ├── base
│   │   └── service.yaml
│   ├── production
│   │   └── deployment.yaml
│   └── staging
│       └── deployment.yaml
├── app.py
├── Dockerfile
├── README.md
├── requirements.txt
├── test_pipeline.py
└── train.py

## Setup and Execution

This project requires a one-time setup of the cloud infrastructure and repository configuration.

1.  **Prerequisites**: `gcloud` CLI, `git`, `python`, `docker`.
2.  **Google Cloud Setup**:
    -   Enable the required APIs (GKE, Artifact Registry, Compute Engine).
    -   Create two GKE clusters (`staging-cluster`, `production-cluster`).
    -   Create a Google Artifact Registry repository to store Docker images.
    -   Create a dedicated IAM Service Account with the necessary permissions.
3.  **GitHub Configuration**:
    -   Create a service account key and add its contents as a `GCP_CREDENTIALS` repository secret.
    -   Configure repository variables for cluster names, locations, and project ID.
4.  **Workflow Execution**:
    -   Follow the CI/CD workflow described above by creating feature branches, opening pull requests to `dev`, and finally merging to `main` for the production release.