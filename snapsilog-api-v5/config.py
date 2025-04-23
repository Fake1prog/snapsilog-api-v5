import os

class Config:
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')  # Use environment variables for security
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')  # Default region is 'us-east-1'
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')  # Name of your S3 bucket
    MODEL_KEY = os.environ.get('S3_MODEL_KEY')  # Path to your model in S3, e.g., 'models/snapsilog-model.pth'
