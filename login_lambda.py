import json
import boto3
import base64
import time

s3 = boto3.client("s3")

BUCKET = "my-processed-bucket-beeren"
KEY = "users.json"

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
    except:
        return response(400, {"message": "Invalid JSON"})

    username = body.get("username")
    password = body.get("password")

    if not username or not password:
        return response(400, {"message": "Missing username or password"})

    # Load users from S3
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=KEY)
        users = json.loads(obj["Body"].read())
    except Exception as e:
        return response(500, {"message": "Error reading S3", "error": str(e)})

    # Validate user
    user = next((u for u in users["users"]
                 if u["username"] == username and u["password"] == password), None)

    if not user:
        return response(401, {"message": "Invalid username or password"})

    # Create simple session token (Base64)
    expiry = int(time.time()) + 3600  # 1 hour
    session_data = f"{username}:{expiry}"
    token = base64.b64encode(session_data.encode()).decode()

    return response(200, {
        "message": "Login successful",
        "token": token,
        "expiresAt": expiry,
        "user": {
            "username": user["username"],
            "fullName": user["fullName"],
            "role": user["role"]
        }
    })

def response(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body)
    }
