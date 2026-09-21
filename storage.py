import os

import boto3
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")  # set for R2/Spaces/MinIO; leave unset for AWS S3
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

DATASET_PREFIX = "dataset/"
MODEL_PREFIX = "model/"
MODEL_FILENAME = "face_cnn.keras"
LABELS_FILENAME = "labels.json"


def get_client():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET_NAME environment variable is not set.")
    kwargs = {"region_name": S3_REGION}
    if S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = S3_ENDPOINT_URL
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are picked up automatically by boto3
    # from the environment if set; no need to pass them explicitly here.
    return boto3.client("s3", **kwargs)


def object_exists(key: str) -> bool:
    client = get_client()
    try:
        client.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def upload_bytes(data: bytes, key: str):
    client = get_client()
    client.put_object(Bucket=S3_BUCKET, Key=key, Body=data)


def upload_file(local_path: str, key: str):
    client = get_client()
    client.upload_file(local_path, S3_BUCKET, key)


def download_file(key: str, local_path: str) -> bool:
    """Returns True if downloaded, False if the key doesn't exist."""
    if not object_exists(key):
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client = get_client()
    client.download_file(S3_BUCKET, key, local_path)
    return True


def list_keys(prefix: str):
    client = get_client()
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# --- App-specific helpers -------------------------------------------------

def upload_dataset_image(person_folder: str, filename: str, image_bytes: bytes):
    key = f"{DATASET_PREFIX}{person_folder}/{filename}"
    upload_bytes(image_bytes, key)


def count_dataset_images(person_folder: str) -> int:
    prefix = f"{DATASET_PREFIX}{person_folder}/"
    return len(list_keys(prefix))


def download_dataset_to_local(local_dataset_dir: str):
    """Mirror every dataset image from S3 into a local folder structure for training."""
    keys = list_keys(DATASET_PREFIX)
    for key in keys:
        relative = key[len(DATASET_PREFIX):]  # "<person>/img_001.jpg"
        if not relative:
            continue
        local_path = os.path.join(local_dataset_dir, relative)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client = get_client()
        client.download_file(S3_BUCKET, key, local_path)


def upload_model_files(local_model_dir: str):
    model_path = os.path.join(local_model_dir, MODEL_FILENAME)
    labels_path = os.path.join(local_model_dir, LABELS_FILENAME)
    if os.path.isfile(model_path):
        upload_file(model_path, f"{MODEL_PREFIX}{MODEL_FILENAME}")
    if os.path.isfile(labels_path):
        upload_file(labels_path, f"{MODEL_PREFIX}{LABELS_FILENAME}")


def download_model_files(local_model_dir: str) -> bool:
    """Pull the latest trained model + labels down from S3. Returns True if both were found."""
    model_ok = download_file(f"{MODEL_PREFIX}{MODEL_FILENAME}", os.path.join(local_model_dir, MODEL_FILENAME))
    labels_ok = download_file(f"{MODEL_PREFIX}{LABELS_FILENAME}", os.path.join(local_model_dir, LABELS_FILENAME))
    return model_ok and labels_ok
