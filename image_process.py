import json
import boto3
import base64

s3 = boto3.client("s3")
rekognition = boto3.client("rekognition")

def lambda_handler(event, context):
    try:
        # Debug print (optional)
        print("EVENT:", json.dumps(event))

        # Extract params safely
        params = {}

        # HTTP API (GET)
        if "queryStringParameters" in event and event["queryStringParameters"]:
            params = event["queryStringParameters"]

        # REST API (GET)
        elif event.get("httpMethod") == "GET":
            params = event.get("queryStringParameters") or {}

        # POST body
        elif event.get("body"):
            params = json.loads(event["body"])

        # Default bucket + key
        bucket = "my-processed-bucket-beeren"
        key = params.get("key")   # fallback
        print("KEY:", key)

        # Read image
        s3_obj = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = s3_obj["Body"].read()

        # Base64 encode
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Rekognition
        rekog_response = rekognition.detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=10,
            MinConfidence=40
        )

        labels = [
            {"name": l["Name"], "confidence": round(l["Confidence"], 2)}
            for l in rekog_response["Labels"]
        ]

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*"
            },
            "body": json.dumps({
                "labels": labels,
                "image_base64": image_base64,
                "content_type": "image/jpeg",
                "key_used": key
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }
