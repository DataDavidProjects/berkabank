from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Google Cloud Services
        "google-cloud-bigquery==3.21.0",
        "google-cloud-aiplatform==1.49.0",
        "google-cloud-storage==2.16.0",
        "google-cloud-artifact-registry==1.11.3",
        "google-cloud-pipeline-components==2.14.0",
        "google-cloud-logging==3.10.0",
        "kfp==2.7.0",
        # Miscellaneous
        "pandas==2.2.2",
        "scikit-learn==1.4.2",
        "numpy==1.26.4",
        "joblib==1.4.0",
        "pyyaml==6.0.1",
        "python-dotenv==1.0.1",
        "pyarrow==16.0.0",
        "probatus==3.1.0",
        "flask==3.0.3",
        "statsmodels==0.14.2",
        # Documentation
        "mkdocs==1.6.0",
        # Code Formatting
        "black==24.4.2",
        "pre-commit==3.7.0",
    ],
)
