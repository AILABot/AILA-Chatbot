"""
S3 Utilities — Client Init • Bucket Create • DOCX Upload • Presigned Download
=============================================================================

Purpose
-------
Small helper module for interacting with Amazon S3:
- Initialize an S3 client with Signature V4
- Create a bucket
- Upload a DOCX (sets ContentType + ContentDisposition)
- Generate a presigned URL for downloads

Configuration (from `backend.database.config.config.settings`)
--------------------------------------------------------------
- AWS_ACCESS_KEY : Access key ID
- AWS_SECRET_KEY : Secret access key
- REGION         : AWS region (e.g., "eu-central-1")
- BUCKET_NAME    : Target S3 bucket

Security Notes
--------------
- Do NOT log or print credentials in production. The current `print(settings.AWS_ACCESS_KEY, settings.AWS_SECRET_KEY)`
  line is suitable only for local debugging and should be removed or guarded via environment flags.
- Presigned URLs grant temporary access; choose sensible expirations and never expose bucket names/keys unnecessarily.

Caveats
-------
- Bucket creation requirements vary by region; some regions require a `CreateBucketConfiguration` with a
  `LocationConstraint`. This helper uses the client's default region—adjust if needed for your deployment.
"""

import boto3,botocore
from backend.database.config.config import settings

def get_client():
    """
    Initialize and return a low-level S3 client configured for Signature V4.

    Uses:
        - settings.AWS_ACCESS_KEY
        - settings.AWS_SECRET_KEY
        - settings.REGION

    Returns:
        botocore.client.S3: An S3 client ready for bucket and object operations.

    Raises:
        botocore.exceptions.NoCredentialsError
        botocore.exceptions.PartialCredentialsError
        botocore.exceptions.ClientError

    Security:
        Avoid logging credentials. The print below is for debugging only.
    """
    boto3.setup_default_session() 
    print(settings.AWS_ACCESS_KEY,settings.AWS_SECRET_KEY)
    s3_client = boto3.client('s3',
                             aws_access_key_id=settings.AWS_ACCESS_KEY,
                             aws_secret_access_key=settings.AWS_SECRET_KEY,
                             region_name=settings.REGION,
                             config=botocore.config.Config(signature_version="s3v4"),)
    return s3_client

def create_s3_bucket(s3_client):
    """
    Create an S3 bucket using the configured bucket name.

    Args:
        s3_client (botocore.client.S3): Client returned by `get_client()`.

    Returns:
        dict: API response from `create_bucket`.

    Notes:
        - Some regions require:
            s3_client.create_bucket(
                Bucket=...,
                CreateBucketConfiguration={"LocationConstraint": settings.REGION}
            )
        - If the bucket already exists and is owned by you, AWS may return an error.
          Handle that at the call site if desired.
    """
    params = {"Bucket": settings.BUCKET_NAME}
    response = s3_client.create_bucket(**params)

### Server-side
def upload(file_path:str,key:str,s3_client):
    """
    Upload a local DOCX file to S3 with explicit headers.

    Args:
        file_path (str): Local path to the file to upload.
        key (str): Object key (destination path/name in the bucket).
        s3_client (botocore.client.S3): Client returned by `get_client()`.

    Behavior:
        - Uses `upload_fileobj` with:
            ContentType       = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ContentDisposition= 'attachment; filename="{file_path}"'

    Returns:
        None

    Caveats:
        - This hard-codes the DOCX MIME type; if you upload other formats, ContentType may be inaccurate.
        - Ensure `settings.BUCKET_NAME` exists and the credentials have `s3:PutObject` permission.
    """
    DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    with open(file_path, "rb") as f:
        f.seek(0)  # make sure
        s3_client.upload_fileobj(
            f, settings.BUCKET_NAME, key,
            ExtraArgs={
                "ContentType": DOCX_MIME,
                "ContentDisposition": f'attachment; filename="{file_path}"'
            }
        )

def download(key:str,s3_client,expires:int=3600):
    """
    Generate a presigned URL for downloading an object.

    Args:
        key (str): Object key in the bucket.
        s3_client (botocore.client.S3): Client returned by `get_client()`.
        expires (int, optional): URL expiration in seconds (default: 3600).

    Returns:
        str: A presigned URL that allows temporary GET access.

    Notes:
        - Requires `s3:GetObject` permission.
        - Share presigned URLs over TLS only; they grant access until expiry.
    """
    response = s3_client.generate_presigned_url(
        'get_object',
        Params = {
            'Bucket': settings.BUCKET_NAME,
            'Key': key
        },
        ExpiresIn = expires
    )
    return response
