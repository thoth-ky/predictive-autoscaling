"""
S3 Utilities
Helper functions for uploading/downloading data to/from S3 for SageMaker training.
"""

import boto3
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict
from io import BytesIO, StringIO


class S3DataManager:
    """
    Manage data uploads/downloads to/from S3 for SageMaker training.
    """

    def __init__(self, bucket_name: str = 'predictive-autoscaling',
                 region: str = 'us-east-1'):
        """
        Initialize S3 manager.

        Args:
            bucket_name: S3 bucket name
            region: AWS region
        """
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.s3_resource = boto3.resource('s3', region_name=region)

    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} already exists")
        except:
            print(f"Creating bucket {self.bucket_name}")
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )

    def upload_dataframe(self, df: pd.DataFrame, s3_path: str, format: str = 'csv'):
        """
        Upload DataFrame to S3.

        Args:
            df: DataFrame to upload
            s3_path: S3 path (e.g., 'data/cpu/train/data.csv')
            format: Format ('csv' or 'parquet')
        """
        if format == 'csv':
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_path,
                Body=csv_buffer.getvalue()
            )
        elif format == 'parquet':
            parquet_buffer = BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_path,
                Body=parquet_buffer.getvalue()
            )
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Uploaded to s3://{self.bucket_name}/{s3_path}")

    def download_dataframe(self, s3_path: str, format: str = 'csv') -> pd.DataFrame:
        """
        Download DataFrame from S3.

        Args:
            s3_path: S3 path
            format: Format ('csv' or 'parquet')

        Returns:
            DataFrame
        """
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_path)

        if format == 'csv':
            return pd.read_csv(BytesIO(obj['Body'].read()))
        elif format == 'parquet':
            return pd.read_parquet(BytesIO(obj['Body'].read()))
        else:
            raise ValueError(f"Unknown format: {format}")

    def upload_numpy_array(self, array: np.ndarray, s3_path: str):
        """
        Upload numpy array to S3.

        Args:
            array: Numpy array
            s3_path: S3 path (e.g., 'data/cpu/train/X.npy')
        """
        buffer = BytesIO()
        np.save(buffer, array)
        buffer.seek(0)

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_path,
            Body=buffer.getvalue()
        )

        print(f"Uploaded array to s3://{self.bucket_name}/{s3_path}")

    def download_numpy_array(self, s3_path: str) -> np.ndarray:
        """
        Download numpy array from S3.

        Args:
            s3_path: S3 path

        Returns:
            Numpy array
        """
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_path)
        buffer = BytesIO(obj['Body'].read())
        return np.load(buffer)

    def upload_training_data(self, metric_name: str,
                            X_train: np.ndarray, y_train_dict: Dict[int, np.ndarray],
                            X_val: np.ndarray, y_val_dict: Dict[int, np.ndarray],
                            X_test: Optional[np.ndarray] = None,
                            y_test_dict: Optional[Dict[int, np.ndarray]] = None):
        """
        Upload complete training dataset to S3 in SageMaker format.

        Args:
            metric_name: Name of metric
            X_train, y_train_dict: Training data
            X_val, y_val_dict: Validation data
            X_test, y_test_dict: Test data (optional)
        """
        print(f"\nUploading training data for {metric_name}...")

        base_path = f"data/{metric_name}"

        # Upload training data
        self.upload_numpy_array(X_train, f"{base_path}/train/X.npy")
        for horizon, y in y_train_dict.items():
            self.upload_numpy_array(y, f"{base_path}/train/y/y_{horizon}.npy")

        # Upload validation data
        self.upload_numpy_array(X_val, f"{base_path}/validation/X.npy")
        for horizon, y in y_val_dict.items():
            self.upload_numpy_array(y, f"{base_path}/validation/y/y_{horizon}.npy")

        # Upload test data if provided
        if X_test is not None:
            self.upload_numpy_array(X_test, f"{base_path}/test/X.npy")
            for horizon, y in y_test_dict.items():
                self.upload_numpy_array(y, f"{base_path}/test/y/y_{horizon}.npy")

        print(f"Data upload complete!")
        print(f"  S3 URI: s3://{self.bucket_name}/{base_path}/")

        return f"s3://{self.bucket_name}/{base_path}/"

    def upload_model(self, model_path: str, s3_path: str):
        """
        Upload trained model to S3.

        Args:
            model_path: Local path to model file
            s3_path: S3 path
        """
        self.s3_client.upload_file(model_path, self.bucket_name, s3_path)
        print(f"Uploaded model to s3://{self.bucket_name}/{s3_path}")

    def download_model(self, s3_path: str, local_path: str):
        """
        Download model from S3.

        Args:
            s3_path: S3 path
            local_path: Local path to save
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3_client.download_file(self.bucket_name, s3_path, local_path)
        print(f"Downloaded model from s3://{self.bucket_name}/{s3_path}")

    def list_metric_data(self, metric_name: str):
        """
        List available data files for a metric.

        Args:
            metric_name: Metric name

        Returns:
            List of S3 keys
        """
        prefix = f"data/{metric_name}/"
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )

        if 'Contents' not in response:
            return []

        return [obj['Key'] for obj in response['Contents']]

    def get_s3_uri(self, s3_path: str) -> str:
        """Get full S3 URI."""
        return f"s3://{self.bucket_name}/{s3_path}"


def prepare_data_for_sagemaker(metric_name: str,
                               X_train: np.ndarray, y_train_dict: Dict,
                               X_val: np.ndarray, y_val_dict: Dict,
                               X_test: Optional[np.ndarray] = None,
                               y_test_dict: Optional[Dict] = None,
                               bucket_name: str = 'predictive-autoscaling') -> str:
    """
    Convenience function to upload data to S3 for SageMaker training.

    Args:
        metric_name: Metric name
        X_train, y_train_dict: Training data
        X_val, y_val_dict: Validation data
        X_test, y_test_dict: Test data
        bucket_name: S3 bucket

    Returns:
        S3 URI for the data
    """
    manager = S3DataManager(bucket_name=bucket_name)
    manager.create_bucket_if_not_exists()

    return manager.upload_training_data(
        metric_name, X_train, y_train_dict,
        X_val, y_val_dict, X_test, y_test_dict
    )


if __name__ == '__main__':
    # Test S3 utilities
    print("S3 Utilities")
    print("=" * 60)

    # Note: Requires AWS credentials configured
    print("\nThis module requires AWS credentials to be configured.")
    print("Configure with: aws configure")

    print("\nExample usage:")
    print("""
# Upload training data
from src.sagemaker.utils.s3_utils import prepare_data_for_sagemaker

s3_uri = prepare_data_for_sagemaker(
    metric_name='cpu',
    X_train=X_train, y_train_dict=y_train_dict,
    X_val=X_val, y_val_dict=y_val_dict
)

# Use with SageMaker estimator
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='src/sagemaker/scripts',
    role=sagemaker_role,
    instance_type='ml.p3.2xlarge',
    framework_version='2.0.0'
)

estimator.fit({
    'train': f'{s3_uri}train/',
    'validation': f'{s3_uri}validation/'
})
    """)
