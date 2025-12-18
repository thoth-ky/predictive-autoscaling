"""
SageMaker Module
AWS SageMaker integration for cloud training and deployment.
"""

from src.sagemaker.utils.s3_utils import (
    upload_data_to_s3,
    download_data_from_s3,
    upload_model_to_s3,
    download_model_from_s3,
)

__all__ = [
    "upload_data_to_s3",
    "download_data_from_s3",
    "upload_model_to_s3",
    "download_model_from_s3",
]
