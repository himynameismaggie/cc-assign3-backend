import hashlib
import json
import logging
import os
from urllib.error import HTTPError
from urllib.parse import unquote_plus
from urllib.request import Request, urlopen

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


logger = logging.getLogger()
logger.setLevel(logging.INFO)

session = boto3.Session()
s3 = session.client("s3")
rekognition = session.client("rekognition")

region = os.environ["AWS_REGION"]
endpoint = os.environ["OPENSEARCH_ENDPOINT"].rstrip("/")


def put_photo_document(document):
    """Write one photo record using the Lambda execution role."""
    object_reference = f"{document['bucket']}/{document['objectKey']}"
    document_id = hashlib.sha256(object_reference.encode("utf-8")).hexdigest()
    url = f"{endpoint}/photos/_doc/{document_id}"
    body = json.dumps(document).encode("utf-8")

    signed_request = AWSRequest(
        method="PUT",
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    credentials = session.get_credentials().get_frozen_credentials()
    SigV4Auth(credentials, "es", region).add_auth(signed_request)

    request = Request(
        url,
        data=body,
        headers=dict(signed_request.headers.items()),
        method="PUT",
    )

    try:
        with urlopen(request, timeout=8) as response:
            response.read()
    except HTTPError as error:
        detail = error.read(1024).decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenSearch PUT failed ({error.code}): {detail}") from error

    return document_id


def lambda_handler(event, context):
    indexed_count = 0

    # S3's setup test event has no Records array.
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        logger.info("Processing photo in bucket=%s key=%s", bucket, key)

        head = s3.head_object(Bucket=bucket, Key=key)
        custom_label_text = head.get("Metadata", {}).get("customlabels", "")
        custom_labels = [
            label.strip() for label in custom_label_text.split(",") if label.strip()
        ]

        detected = rekognition.detect_labels(
            Image={"S3Object": {"Bucket": bucket, "Name": key}}
        )
        automatic_labels = [label["Name"] for label in detected.get("Labels", [])]
        labels = list(dict.fromkeys(custom_labels + automatic_labels))

        document_id = put_photo_document({
            "objectKey": key,
            "bucket": bucket,
            "createdTimestamp": record["eventTime"],
            "labels": labels,
        })
        indexed_count += 1
        logger.info(
            "Indexed document=%s custom_labels=%d rekognition_labels=%d",
            document_id,
            len(custom_labels),
            len(automatic_labels),
        )

    return {"indexed": indexed_count}
